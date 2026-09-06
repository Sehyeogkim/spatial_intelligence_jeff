#!/usr/bin/env python3
"""Run a World2Work coffee-delivery episode with NVIDIA Nova Carter.

The learned tabular policy is exported as a route by train_nav_qlearning.py.
This script consumes that route, follows it with a differential-drive
controller, and writes 10 Hz robot state/action records as JSONL.

Isaac Sim must be launched with its bundled Python.  ``--dry-run`` uses the
same route follower and logger with deterministic unicycle kinematics, so the
route/schema can be checked on a laptop before spending GPU time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "world2work.robot-episode-jsonl.v1"


def _project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir.parent, script_dir, Path("/workspace")]
    for candidate in candidates:
        if (candidate / "config" / "corgi_cafe_task.json").is_file():
            return candidate
    return script_dir.parent


PROJECT_ROOT = _project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a learned Marble-grid route with Nova Carter."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "corgi_cafe_task.json",
    )
    parser.add_argument("--target", default=None, help="For example: table_7")
    parser.add_argument("--route", type=Path, default=None)
    parser.add_argument("--grid-meta", type=Path, default=None)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this waypoint in the exported route (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="A .jsonl path or an output directory.",
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--robot-usd", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Optional MP4 path. Rendering stays disabled when this is omitted.",
    )
    parser.add_argument("--disable-marble-visual", action="store_true")
    parser.add_argument(
        "--marble-source",
        choices=["segmented", "fused"],
        default=None,
        help=(
            "segmented = per-class/per-table GLB from segment_marble_collider.py; "
            "fused = raw World Labs collider. Default: environment.marble_source "
            "in the config, falling back to fused when the segmented GLB is missing."
        ),
    )
    parser.add_argument("--enable-marble-collision", action="store_true")
    parser.add_argument("--enable-proxies", action="store_true")
    parser.add_argument("--no-route-markers", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_path(value: str | os.PathLike[str], base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _float_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a two-number array")
    if len(value) < 2:
        raise ValueError(f"{label} must contain at least two values")
    return float(value[0]), float(value[1])


def _cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    return int(value[0]), int(value[1])


def path_length(points: Sequence[tuple[float, float]]) -> float:
    return sum(
        math.hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )


@dataclass(frozen=True)
class RouteData:
    source: str
    waypoints: list[tuple[float, float]]
    cells: list[tuple[int, int] | None]
    target_xy: tuple[float, float]
    path_length_m: float
    raw: dict[str, Any]


def slice_route(route: RouteData, start_index: int) -> RouteData:
    if start_index < 0 or start_index >= len(route.waypoints) - 1:
        raise ValueError(
            f"--start-index must be in [0, {len(route.waypoints) - 2}], "
            f"got {start_index}"
        )
    if start_index == 0:
        return route
    waypoints = route.waypoints[start_index:]
    cells = route.cells[start_index:]
    return RouteData(
        source=f"{route.source}#start_index={start_index}",
        waypoints=waypoints,
        cells=cells,
        target_xy=route.target_xy,
        path_length_m=path_length(waypoints),
        raw={**route.raw, "evaluation_start_index": start_index},
    )


def shortest_distance_m(route: RouteData, grid_meta: dict[str, Any], target_id: str) -> float:
    declared = route.raw.get("shortest_path_length_m")
    if declared is not None:
        return max(0.0, float(declared))
    target_meta = grid_meta.get("targets", {}).get(target_id, {})
    steps = target_meta.get("shortest_path_steps")
    cell_size = grid_meta.get("cell_size_m")
    if steps is not None and cell_size is not None:
        # If the route was sliced, its remaining polyline is the appropriate
        # shortest-distance denominator for that held-out starting waypoint.
        full_length = float(steps) * float(cell_size)
        return min(full_length, route.path_length_m)
    return route.path_length_m


def cells_to_waypoints(
    cells: Sequence[Any], grid_meta: dict[str, Any]
) -> list[tuple[float, float]]:
    origin = _float_pair(
        grid_meta.get("origin_xy_m", grid_meta.get("origin")),
        "grid_meta.origin_xy_m",
    )
    cell_size = float(
        grid_meta.get(
            "cell_size_m",
            grid_meta.get("cell_size", grid_meta.get("resolution_m", 0.0)),
        )
    )
    if cell_size <= 0:
        raise ValueError("grid_meta.cell_size_m must be positive")
    result: list[tuple[float, float]] = []
    for index, value in enumerate(cells):
        parsed = _cell(value)
        if parsed is None:
            raise ValueError(f"route cell {index} is invalid: {value!r}")
        row, col = parsed
        # Stable grid contract: row increases +Y and col increases +X.
        result.append(
            (
                origin[0] + (col + 0.5) * cell_size,
                origin[1] + (row + 0.5) * cell_size,
            )
        )
    return result


def load_route(
    route_path: Path,
    grid_meta: dict[str, Any],
    config: dict[str, Any],
    target_id: str,
) -> RouteData:
    if route_path.is_file():
        raw = load_json(route_path)
        raw_waypoints = raw.get("waypoints_xy_m")
        raw_cells = raw.get("cells", raw.get("route_cells", []))
        if raw_waypoints:
            waypoints = [
                _float_pair(value, f"waypoints_xy_m[{index}]")
                for index, value in enumerate(raw_waypoints)
            ]
        elif raw_cells:
            waypoints = cells_to_waypoints(raw_cells, grid_meta)
        else:
            raise ValueError(
                f"{route_path} has neither waypoints_xy_m nor cells"
            )
        cells = [_cell(value) for value in raw_cells]
        if len(cells) != len(waypoints):
            cells = [None] * len(waypoints)
        target = raw.get("target", {})
        if isinstance(target, dict) and target.get("xy_m") is not None:
            target_xy = _float_pair(target["xy_m"], "route.target.xy_m")
        else:
            target_xy = waypoints[-1]
        source = str(route_path)
        declared_length = raw.get("path_length_m")
        length = (
            float(declared_length)
            if declared_length is not None
            else path_length(waypoints)
        )
    else:
        nav = config["navigation"]
        targets = config.get("targets", {})
        if target_id not in targets:
            raise FileNotFoundError(
                f"Route is missing and target has no fallback: {route_path}"
            )
        start = _float_pair(nav["fallback_start_xy_m"], "fallback_start_xy_m")
        target_pose = targets[target_id]["fallback_pose_xyz_m"]
        target_xy = _float_pair(target_pose, f"targets.{target_id}")
        waypoints = [start, target_xy]
        cells = [None, None]
        length = path_length(waypoints)
        source = f"fallback_direct:{route_path}"
        raw = {
            "schema_version": "world2work.nav-route.fallback.v1",
            "target_id": target_id,
            "waypoints_xy_m": [list(point) for point in waypoints],
            "warning": "Generated route was unavailable; direct fallback used.",
        }
        print(
            f"ROUTE_WARNING missing={route_path} using_direct_fallback=true",
            flush=True,
        )

    if len(waypoints) < 2:
        raise ValueError("A route must contain at least two waypoints")
    for index, point in enumerate(waypoints):
        if not all(math.isfinite(component) for component in point):
            raise ValueError(f"Waypoint {index} is not finite: {point}")
    return RouteData(source, waypoints, cells, target_xy, length, raw)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yaw_to_wxyz(yaw: float) -> list[float]:
    return [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]


def wxyz_to_yaw(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wxyz_to_xyzw(quaternion: Sequence[float]) -> list[float]:
    return [
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
        float(quaternion[0]),
    ]


def rotated_offset_pose(
    base_position: Sequence[float],
    base_yaw: float,
    offset_xyz: Sequence[float],
) -> tuple[list[float], list[float]]:
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    ox, oy, oz = (float(value) for value in offset_xyz)
    x = float(base_position[0]) + cos_yaw * ox - sin_yaw * oy
    y = float(base_position[1]) + sin_yaw * ox + cos_yaw * oy
    z = float(base_position[2]) + oz
    return [x, y, z], wxyz_to_xyzw(yaw_to_wxyz(base_yaw))


class RouteFollower:
    def __init__(self, route: RouteData, config: dict[str, Any]) -> None:
        self.route = route
        self.nav = config["navigation"]
        self.success = config["success"]
        self.waypoint_index = 1
        self.arrival_started_s: float | None = None

    def _advance_near_waypoints(self, x: float, y: float) -> None:
        radius = float(self.nav["waypoint_radius_m"])
        while self.waypoint_index < len(self.route.waypoints) - 1:
            waypoint = self.route.waypoints[self.waypoint_index]
            if math.hypot(waypoint[0] - x, waypoint[1] - y) > radius:
                break
            self.waypoint_index += 1

    def _lookahead_index(self, x: float, y: float) -> int:
        lookahead = float(self.nav["lookahead_m"])
        index = self.waypoint_index
        # Select the furthest future route point still within lookahead.  This
        # smooths dense 10 cm grid paths without changing the learned route.
        while index + 1 < len(self.route.waypoints):
            candidate = self.route.waypoints[index + 1]
            if math.hypot(candidate[0] - x, candidate[1] - y) > lookahead:
                break
            index += 1
        return index

    def command(
        self, x: float, y: float, yaw: float, elapsed_s: float
    ) -> tuple[float, float, str, bool]:
        self._advance_near_waypoints(x, y)
        goal_error = math.hypot(
            self.route.target_xy[0] - x, self.route.target_xy[1] - y
        )
        if goal_error <= float(self.success["target_radius_m"]):
            if self.arrival_started_s is None:
                self.arrival_started_s = elapsed_s
            settled = elapsed_s - self.arrival_started_s >= float(
                self.success["settle_time_s"]
            )
            return 0.0, 0.0, "DONE" if settled else "ARRIVE", settled
        self.arrival_started_s = None

        waypoint = self.route.waypoints[self._lookahead_index(x, y)]
        dx, dy = waypoint[0] - x, waypoint[1] - y
        distance = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        heading_error = wrap_angle(desired_yaw - yaw)
        max_linear = float(self.nav["linear_speed_max_mps"])
        max_angular = float(self.nav["angular_speed_max_radps"])
        angular = clamp(
            float(self.nav["heading_kp"]) * heading_error,
            -max_angular,
            max_angular,
        )
        if abs(heading_error) >= float(self.nav["turn_in_place_threshold_rad"]):
            linear = 0.0
        else:
            slowdown = max(float(self.nav["slowdown_distance_m"]), 1e-6)
            heading_scale = max(
                float(self.nav["minimum_drive_heading_cosine"]),
                math.cos(heading_error),
            )
            linear = max_linear * min(1.0, distance / slowdown) * heading_scale
        return linear, angular, "NAVIGATE", False

    def policy_cell(self) -> list[int] | None:
        index = min(self.waypoint_index, len(self.route.cells) - 1)
        cell = self.route.cells[index] if self.route.cells else None
        return list(cell) if cell is not None else None


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        self.logged_steps = 0

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.handle.flush()
        if record.get("record_type") == "step":
            self.logged_steps += 1

    def close(self) -> None:
        self.handle.close()


def output_paths(args: argparse.Namespace, episode_id: str) -> tuple[Path, Path]:
    if args.output.suffix.lower() == ".jsonl":
        jsonl_path = args.output
    else:
        jsonl_path = args.output / f"episode_{episode_id}.jsonl"
    summary_path = args.summary or jsonl_path.with_suffix(".summary.json")
    return jsonl_path, summary_path


def environment_transform(
    config: dict[str, Any], grid_meta: dict[str, Any]
) -> tuple[float, float, float]:
    configured = config["environment"]["transform"]
    transform = grid_meta.get("transform", {})
    scale = float(
        transform.get("scale", configured.get("metric_scale_factor", 1.0))
    )
    ground = float(
        transform.get(
            "ground_offset", configured.get("ground_plane_offset_m", 0.0)
        )
    )
    visual_bias = float(configured.get("visual_z_bias_m", 0.0))
    return scale, ground, visual_bias


SEMANTIC_CLASSES = (
    "floor",
    "ceiling",
    "wall",
    "furniture",
    "seat",
    "table_top",
    "other",
    "outside",
)


def marble_source_paths(
    config: dict[str, Any], override: str | None
) -> tuple[str, Path, Path]:
    """Pick the GLB that becomes the Marble scene and its cached USD."""
    env = config["environment"]
    requested = override or str(env.get("marble_source", "segmented"))
    fused_glb = resolve_path(env["collider_glb"])
    fused_usd = resolve_path(env["converted_usd"])
    if requested == "segmented":
        segmented = env.get("segmented_glb")
        if segmented and resolve_path(segmented).is_file():
            usd = env.get(
                "segmented_usd", str(fused_usd.with_name("segmented_isaac.usd"))
            )
            return "segmented", resolve_path(segmented), resolve_path(usd)
        print(
            "MARBLE_SOURCE_FALLBACK segmented GLB missing "
            f"({segmented}); run scripts/segment_marble_collider.py. Using fused collider.",
            flush=True,
        )
    return "fused", fused_glb, fused_usd


def prim_semantic_class(name: str) -> tuple[str | None, str | None]:
    """Map a segmented.glb node/prim name to (class, instance)."""
    base = name.split("/")[-1]
    if base in SEMANTIC_CLASSES:
        return base, None
    stem, _, number = base.rpartition("_")
    if stem == "table" and number.isdigit():
        return "table_top", base
    if stem == "seat" and number.isdigit():
        return "seat", base
    # The asset converter sometimes suffixes duplicates (table_3_1) or wraps
    # the mesh in a same-named Xform; strip one trailing numeric token.
    if number.isdigit():
        return prim_semantic_class(stem)
    return None, None


def load_semantic_scene(
    config: dict[str, Any], target_id: str, grid_meta: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Summarise semantics.json for the episode_start record (no Isaac needed).

    The physical table behind ``target_id`` comes from grid_meta first (written
    by marble_to_grid.py as targets.<id>.semantic_instance), then from the task
    config, then from a detected table that happens to share the id.
    """
    env = config["environment"]
    path_value = env.get("semantics_json")
    if not path_value:
        return None
    path = resolve_path(path_value)
    if not path.is_file():
        return None
    data = load_json(path)
    tables = {table["name"]: table for table in data.get("tables", [])}
    grid_target = (grid_meta or {}).get("targets", {}).get(target_id, {})
    target_cfg = config.get("targets", {}).get(target_id, {})
    instance = grid_target.get("semantic_instance") or target_cfg.get("semantic_instance")
    if instance is None and target_id in tables:
        instance = target_id
    table = tables.get(instance) if instance else None
    return {
        "source": str(path),
        "schema_version": data.get("schema_version"),
        "table_count": len(tables),
        "seat_count": len(data.get("seats", [])),
        "class_triangles": {
            name: int(info.get("triangles", 0))
            for name, info in data.get("classes", {}).items()
        },
        "target_instance": instance,
        "target_selection": grid_target.get("selection"),
        "target_approach_gap_m": grid_target.get("approach_gap_m"),
        "target_table_pose_xyz_m": (
            [float(v) for v in table["centroid_xyz_m"]] if table else None
        ),
        "target_table_height_m": (
            float(table["height_z_m"]) if table else None
        ),
    }


def apply_semantic_label(prim: Any, class_name: str, instance: str | None) -> bool:
    """Attach class/instance semantics using whichever Isaac API is available."""
    try:
        from isaacsim.core.utils.semantics import add_labels  # Isaac Sim >= 5.0

        add_labels(prim, labels=[class_name], instance_name="class")
        if instance:
            add_labels(prim, labels=[instance], instance_name="instance")
        return True
    except Exception:
        pass
    try:
        from isaacsim.core.utils.semantics import add_update_semantics  # Isaac Sim 4.x

        add_update_semantics(prim, semantic_label=class_name, type_label="class")
        if instance:
            add_update_semantics(
                prim, semantic_label=instance, type_label="instance", suffix="_inst"
            )
        return True
    except Exception:
        pass
    try:
        from pxr import Semantics

        api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        api.CreateSemanticTypeAttr().Set("class")
        api.CreateSemanticDataAttr().Set(class_name)
        if instance:
            inst = Semantics.SemanticsAPI.Apply(prim, "Semantics_inst")
            inst.CreateSemanticTypeAttr().Set("instance")
            inst.CreateSemanticDataAttr().Set(instance)
        return True
    except Exception:
        return False


def build_start_record(
    *,
    episode_id: str,
    config: dict[str, Any],
    route: RouteData,
    target_id: str,
    scale: float,
    ground_offset: float,
    marble_collision: bool,
    proxies: bool,
    dry_run: bool,
    asset_usd: str | None,
    wheel_names: Sequence[str],
    wheel_radius: float,
    wheel_base: float,
    semantic_scene: dict[str, Any] | None = None,
    marble_source: str | None = None,
) -> dict[str, Any]:
    env = config["environment"]
    robot = config["robot"]
    environment: dict[str, Any] = {
        "world_id": env["world_id"],
        "world_provider": env["world_provider"],
        "physics_surface": env["physics_surface"],
        "metric_scale_factor": scale,
        "ground_plane_offset_m": ground_offset,
        "marble_collision_enabled": marble_collision,
        "proxy_collision_enabled": proxies,
    }
    if marble_source is not None:
        environment["marble_source"] = marble_source
    if semantic_scene is not None:
        environment["semantic_scene"] = semantic_scene
    return {
        "record_type": "episode_start",
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "task_id": config["task_id"],
        "instruction": config["instruction"],
        "target_id": target_id,
        "environment": environment,
        "robot": {
            "name": robot["name"],
            "mode": robot["mode"],
            "asset_usd": asset_usd,
            "wheel_dof_names": list(wheel_names),
            "wheel_radius_m": wheel_radius,
            "wheel_base_m": wheel_base,
        },
        "route": {
            "source": route.source,
            "waypoint_count": len(route.waypoints),
            "start_xy_m": list(route.waypoints[0]),
            "target_xy_m": list(route.target_xy),
            "path_length_m": route.path_length_m,
        },
        "dry_run": dry_run,
    }


def build_step_record(
    *,
    episode_id: str,
    step_index: int,
    elapsed_s: float,
    phase: str,
    position: Sequence[float],
    orientation_wxyz: Sequence[float],
    linear_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    wheel_velocity: Sequence[float],
    command: tuple[float, float],
    cup_offset: Sequence[float],
    target_id: str,
    target_xyz: Sequence[float],
    policy_cell: list[int] | None,
    waypoint_index: int,
    reward: float,
    done: bool,
    collision_source: str = "not_instrumented",
) -> dict[str, Any]:
    yaw = wxyz_to_yaw(orientation_wxyz)
    cup_position, cup_orientation = rotated_offset_pose(position, yaw, cup_offset)
    xy_error = math.hypot(
        float(target_xyz[0]) - float(position[0]),
        float(target_xyz[1]) - float(position[1]),
    )
    return {
        "record_type": "step",
        "episode_id": episode_id,
        "step_index": step_index,
        "t": round(elapsed_s, 6),
        "phase": phase,
        "base_pose": {
            "position_xyz_m": [float(value) for value in position[:3]],
            "orientation_xyzw": wxyz_to_xyzw(orientation_wxyz),
        },
        "base_twist": {
            "linear_xyz_mps": [float(value) for value in linear_velocity[:3]],
            "angular_xyz_radps": [float(value) for value in angular_velocity[:3]],
        },
        "wheel_velocity_radps": {
            "left": float(wheel_velocity[0]),
            "right": float(wheel_velocity[1]),
        },
        "cmd_vel": {
            "linear_x_mps": float(command[0]),
            "angular_z_radps": float(command[1]),
        },
        "cup_pose": {
            "position_xyz_m": cup_position,
            "orientation_xyzw": cup_orientation,
        },
        "target": {
            "id": target_id,
            "pose_xyz_m": [float(value) for value in target_xyz[:3]],
            "xy_error_m": xy_error,
        },
        "policy_cell": policy_cell,
        "route_waypoint_index": int(waypoint_index),
        "collision": None,
        "collision_detection": collision_source,
        "reward": float(reward),
        "done": bool(done),
    }


def build_outcome(
    *,
    episode_id: str,
    success: bool,
    reason: str,
    elapsed_s: float,
    xy_error: float,
    writer: JsonlWriter,
    follower: RouteFollower,
    total_reward: float,
    dry_run: bool,
    actual_distance_m: float,
    shortest_distance: float,
    collisions: int | None,
) -> dict[str, Any]:
    efficiency = 0.0
    if success and actual_distance_m > 1e-9:
        efficiency = min(1.0, shortest_distance / actual_distance_m)
    return {
        "record_type": "outcome",
        "episode_id": episode_id,
        "success": success,
        "reason": reason,
        "elapsed_s": round(max(0.0, elapsed_s), 6),
        "xy_error_m": max(0.0, float(xy_error)),
        "logged_steps": writer.logged_steps,
        "route_waypoints_reached": min(
            follower.waypoint_index + 1, len(follower.route.waypoints)
        ),
        "route_waypoint_count": len(follower.route.waypoints),
        "total_reward": float(total_reward),
        "actual_distance_m": max(0.0, float(actual_distance_m)),
        "shortest_distance_m": max(0.0, float(shortest_distance)),
        "path_efficiency": float(efficiency),
        "collisions": collisions,
        "dry_run": dry_run,
        "output_jsonl": str(writer.path),
    }


def write_summary(path: Path, start: dict[str, Any], outcome: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"episode": start, "outcome": outcome}, indent=2) + "\n",
        encoding="utf-8",
    )


def wheel_speeds(
    linear: float,
    angular: float,
    wheel_radius: float,
    wheel_base: float,
    signs: Sequence[float],
) -> tuple[float, float]:
    left = (linear - angular * wheel_base * 0.5) / wheel_radius
    right = (linear + angular * wheel_base * 0.5) / wheel_radius
    return left * float(signs[0]), right * float(signs[1])


def initial_yaw(route: RouteData, fallback: float) -> float:
    for first, second in zip(route.waypoints, route.waypoints[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        if math.hypot(dx, dy) > 1e-6:
            return math.atan2(dy, dx)
    return fallback


def target_xyz(
    config: dict[str, Any], route: RouteData, target_id: str
) -> list[float]:
    fallback = config.get("targets", {}).get(target_id, {}).get(
        "fallback_pose_xyz_m", [route.target_xy[0], route.target_xy[1], 0.0]
    )
    return [route.target_xy[0], route.target_xy[1], float(fallback[2])]


def run_dry(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    route: RouteData,
    target_id: str,
    grid_meta: dict[str, Any],
    writer: JsonlWriter,
    summary_path: Path,
    episode_id: str,
) -> dict[str, Any]:
    physics_hz = int(config["logging"]["physics_hz"])
    state_hz = int(config["logging"]["state_hz"])
    if physics_hz <= 0 or state_hz <= 0 or physics_hz % state_hz:
        raise ValueError("physics_hz must be a positive multiple of state_hz")
    log_interval = physics_hz // state_hz
    dt = 1.0 / physics_hz
    timeout = float(args.timeout_s or config["success"]["timeout_s"])
    wheel_radius = float(config["robot"]["wheel_radius_m"])
    wheel_base = float(config["robot"]["wheel_base_m"])
    signs = config["robot"].get("wheel_velocity_signs", [1.0, 1.0])
    wheel_names = config["robot"]["wheel_dof_name_pairs"][0]
    scale, ground_offset, _ = environment_transform(config, grid_meta)
    marble_source, _, _ = marble_source_paths(config, args.marble_source)
    start_record = build_start_record(
        episode_id=episode_id,
        config=config,
        route=route,
        target_id=target_id,
        scale=scale,
        ground_offset=ground_offset,
        marble_collision=False,
        proxies=False,
        dry_run=True,
        asset_usd=None,
        wheel_names=wheel_names,
        wheel_radius=wheel_radius,
        wheel_base=wheel_base,
        semantic_scene=load_semantic_scene(config, target_id, grid_meta),
        marble_source=marble_source,
    )
    writer.write(start_record)
    follower = RouteFollower(route, config)
    x, y = route.waypoints[0]
    z = float(config["robot"]["start_z_m"])
    yaw = initial_yaw(
        route, float(config["navigation"].get("fallback_start_yaw_rad", 0.0))
    )
    target = target_xyz(config, route, target_id)
    cup_offset = config["payload"]["cup"]["offset_xyz_m"]
    previous_error = math.hypot(target[0] - x, target[1] - y)
    shortest_distance = shortest_distance_m(route, grid_meta, target_id)
    actual_distance = 0.0
    total_reward = 0.0
    outcome: dict[str, Any] | None = None
    last_command = (0.0, 0.0)
    last_wheels = (0.0, 0.0)

    for frame in range(int(math.ceil(timeout * physics_hz)) + 1):
        elapsed = frame * dt
        linear, angular, phase, reached = follower.command(x, y, yaw, elapsed)
        if elapsed >= timeout and not reached:
            linear, angular, phase = 0.0, 0.0, "TIMEOUT"
        last_command = (linear, angular)
        last_wheels = wheel_speeds(
            linear, angular, wheel_radius, wheel_base, signs
        )
        old_x, old_y = x, y
        x += linear * math.cos(yaw) * dt
        y += linear * math.sin(yaw) * dt
        actual_distance += math.hypot(x - old_x, y - old_y)
        yaw = wrap_angle(yaw + angular * dt)
        error = math.hypot(target[0] - x, target[1] - y)
        reward = (
            (previous_error - error) * float(config["reward"]["progress_scale"])
            + float(config["reward"]["step_penalty"])
        )
        done = reached or phase == "TIMEOUT"
        if reached:
            reward += float(config["reward"]["success_bonus"])
        elif phase == "TIMEOUT":
            reward += float(config["reward"]["timeout_penalty"])
        total_reward += reward
        previous_error = error

        if frame % log_interval == 0 or done:
            record = build_step_record(
                episode_id=episode_id,
                step_index=writer.logged_steps,
                elapsed_s=elapsed,
                phase=phase,
                position=[x, y, z],
                orientation_wxyz=yaw_to_wxyz(yaw),
                linear_velocity=[linear * math.cos(yaw), linear * math.sin(yaw), 0.0],
                angular_velocity=[0.0, 0.0, angular],
                wheel_velocity=last_wheels,
                command=last_command,
                cup_offset=cup_offset,
                target_id=target_id,
                target_xyz=target,
                policy_cell=follower.policy_cell(),
                waypoint_index=follower.waypoint_index,
                reward=reward,
                done=done,
            )
            writer.write(record)
        if done:
            success = reached
            reason = "target_reached" if reached else "timeout"
            outcome = build_outcome(
                episode_id=episode_id,
                success=success,
                reason=reason,
                elapsed_s=elapsed,
                xy_error=error,
                writer=writer,
                follower=follower,
                total_reward=total_reward,
                dry_run=True,
                actual_distance_m=actual_distance,
                shortest_distance=shortest_distance,
                collisions=0,
            )
            break

    if outcome is None:
        raise RuntimeError("Dry-run loop ended without an outcome")
    writer.write(outcome)
    write_summary(summary_path, start_record, outcome)
    return outcome


def _asset_exists(uri: str) -> bool:
    try:
        import omni.client

        result, _entry = omni.client.stat(uri)
        return result == omni.client.Result.OK
    except Exception:
        return False


def resolve_robot_asset(
    assets_root: str, config: dict[str, Any], override: str | None
) -> str:
    if override:
        candidates = [override]
    else:
        candidates = config["robot"]["asset_candidates"]
    expanded: list[str] = []
    for candidate in candidates:
        value = str(candidate)
        if "://" in value or value.startswith("omniverse:"):
            expanded.append(value)
        elif Path(value).is_absolute() and Path(value).is_file():
            expanded.append(value)
        else:
            expanded.append(assets_root.rstrip("/") + "/" + value.lstrip("/"))
    for candidate in expanded:
        if _asset_exists(candidate) or Path(candidate).is_file():
            return candidate
    # Asset-root references can resolve lazily even when omni.client.stat is not
    # available in a minimal container.  Use the preferred candidate and verify
    # the referenced prim after loading.
    return expanded[0]


def discover_wheel_dofs(
    dof_names: Sequence[str], configured_pairs: Sequence[Sequence[str]]
) -> tuple[str, str, int, int]:
    names = [str(name) for name in dof_names]
    exact = {name.lower(): index for index, name in enumerate(names)}
    for pair in configured_pairs:
        if len(pair) != 2:
            continue
        left_key, right_key = str(pair[0]).lower(), str(pair[1]).lower()
        if left_key in exact and right_key in exact:
            return names[exact[left_key]], names[exact[right_key]], exact[left_key], exact[right_key]

    def scored(side: str) -> list[tuple[int, int, str]]:
        matches: list[tuple[int, int, str]] = []
        for index, name in enumerate(names):
            lowered = name.lower()
            if "wheel" not in lowered or side not in lowered:
                continue
            penalty = 0
            if "caster" in lowered:
                penalty += 100
            if "joint" not in lowered:
                penalty += 5
            matches.append((penalty, len(name), name))
        return sorted(matches)

    left_matches, right_matches = scored("left"), scored("right")
    if not left_matches or not right_matches:
        raise RuntimeError(
            "Could not identify left/right drive wheel DOFs. "
            f"Discovered DOFs: {names}"
        )
    left_name, right_name = left_matches[0][2], right_matches[0][2]
    return left_name, right_name, names.index(left_name), names.index(right_name)


def convert_marble_glb(
    simulation_app: Any, source: Path, destination: Path, single_mesh: bool = True
) -> Path:
    if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    import omni.kit.asset_converter

    async def convert() -> None:
        context = omni.kit.asset_converter.AssetConverterContext()
        context.ignore_materials = False
        context.ignore_animation = True
        context.ignore_cameras = True
        # single_mesh would fuse the per-class/per-table nodes of segmented.glb
        # back into one prim and destroy the labels; keep it only for the raw
        # World Labs collider.
        context.single_mesh = single_mesh
        context.use_meter_as_world_unit = False
        context.create_world_as_default_root_prim = True
        task = omni.kit.asset_converter.get_instance().create_converter_task(
            str(source), str(destination), lambda _progress, _total: None, context
        )
        if not await task.wait_until_finished():
            raise RuntimeError(
                "Marble GLB conversion failed: "
                f"{task.get_status()} {task.get_error_message()}"
            )

    simulation_app.run_coroutine(convert())
    if not destination.is_file():
        raise RuntimeError(f"Asset converter produced no file: {destination}")
    return destination


def add_marble_scene(
    *,
    simulation_app: Any,
    stage: Any,
    config: dict[str, Any],
    scale: float,
    ground_offset: float,
    visual_bias: float,
    enable_collision: bool,
    marble_source: str | None = None,
) -> tuple[Path, int, dict[str, Any]]:
    """Load the Marble scene under /World/MarbleCafe.

    With the segmented GLB every class (floor, wall, table_top, ...) and every
    table instance arrives as its own mesh prim.  Those prims get a display
    colour, a semantic label, and collision only for the classes listed in
    environment.semantic_classes.collision_classes.  The fused collider keeps
    the previous behaviour: one grey mesh, optional whole-mesh collision.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    source_kind, source, destination = marble_source_paths(config, marble_source)
    if not source.is_file():
        raise FileNotFoundError(f"Marble collider is missing: {source}")
    segmented = source_kind == "segmented"
    converted = convert_marble_glb(
        simulation_app, source, destination, single_mesh=not segmented
    )
    root = UsdGeom.Xform.Define(stage, "/World/MarbleCafe")
    xform = UsdGeom.XformCommonAPI(root)
    xform.SetTranslate(Gf.Vec3d(0.0, 0.0, ground_offset + visual_bias))
    xform.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0))
    xform.SetScale(Gf.Vec3f(scale, scale, scale))
    geometry = UsdGeom.Xform.Define(stage, "/World/MarbleCafe/Geometry")
    geometry.GetPrim().GetReferences().AddReference(str(converted))
    for _ in range(120):
        simulation_app.update()
        meshes = [
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith("/World/MarbleCafe")
            and prim.IsA(UsdGeom.Mesh)
        ]
        if meshes:
            break
    if not meshes:
        raise RuntimeError("Converted Marble scene contains no mesh prim")

    semantic_cfg = config["environment"].get("semantic_classes", {})
    collision_classes = set(
        semantic_cfg.get(
            "collision_classes", ["wall", "furniture", "seat", "table_top"]
        )
    )
    hidden_classes = set(semantic_cfg.get("hidden_classes", ["outside"]))
    colours = semantic_cfg.get("display_color_rgb", {})
    default_colour = Gf.Vec3f(0.38, 0.42, 0.46)

    class_counts: dict[str, int] = {}
    collision_prims = 0
    labeled_prims = 0
    unclassified: list[str] = []
    tables: list[str] = []
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        mesh.CreateDoubleSidedAttr(True)
        if not segmented:
            mesh.CreateDisplayColorAttr([default_colour])
            if enable_collision:
                UsdPhysics.CollisionAPI.Apply(prim)
                try:
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
                except Exception:
                    pass
                collision_prims += 1
            continue

        class_name, instance = prim_semantic_class(str(prim.GetPath()))
        if class_name is None:
            # Ask the parent Xform too; the converter may nest Mesh under Xform.
            class_name, instance = prim_semantic_class(str(prim.GetPath().GetParentPath()))
        if class_name is None:
            unclassified.append(str(prim.GetPath()))
            mesh.CreateDisplayColorAttr([default_colour])
            continue
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        if instance and class_name == "table_top":
            tables.append(instance)
        rgb = colours.get(class_name)
        mesh.CreateDisplayColorAttr(
            [Gf.Vec3f(*[float(v) for v in rgb]) if rgb else default_colour]
        )
        if class_name in hidden_classes:
            UsdGeom.Imageable(prim).MakeInvisible()
        if apply_semantic_label(prim, class_name, instance):
            labeled_prims += 1
        if enable_collision and class_name in collision_classes:
            UsdPhysics.CollisionAPI.Apply(prim)
            try:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            except Exception:
                pass
            collision_prims += 1

    summary = {
        "source": source_kind,
        "glb": str(source),
        "mesh_prims": len(meshes),
        "class_prims": class_counts,
        "table_instances": len(tables),
        "labeled_prims": labeled_prims,
        "collision_prims": collision_prims,
        "hidden_classes": sorted(hidden_classes) if segmented else [],
        "unclassified_prims": unclassified[:10],
    }
    if unclassified:
        print(
            f"MARBLE_UNCLASSIFIED_PRIMS count={len(unclassified)} "
            f"sample={unclassified[:5]}",
            flush=True,
        )
    print(f"MARBLE_SCENE {json.dumps(summary, sort_keys=True)}", flush=True)
    return converted, len(meshes), summary


def add_proxy_obstacles(
    world: Any, grid_meta: dict[str, Any]
) -> int:
    import numpy as np
    from isaacsim.core.api.objects import FixedCuboid

    proxies = grid_meta.get("obstacle_proxies", [])
    count = 0
    for index, proxy in enumerate(proxies[:200]):
        center = proxy.get("center_xy_m")
        size = proxy.get("size_xy_m")
        if not center or not size:
            continue
        sx, sy = max(float(size[0]), 0.02), max(float(size[1]), 0.02)
        height = float(proxy.get("height_m", 0.8))
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/CollisionProxies/Proxy_{index:03d}",
                name=f"collision_proxy_{index:03d}",
                position=np.array([float(center[0]), float(center[1]), height * 0.5]),
                scale=np.array([sx, sy, height]),
                color=np.array([0.2, 0.55, 0.85]),
            )
        )
        count += 1
    return count


def add_visuals(
    stage: Any,
    route: RouteData,
    config: dict[str, Any],
    target: Sequence[float],
    include_route: bool,
) -> tuple[Any, Any]:
    from pxr import Gf, Sdf, UsdGeom, UsdLux

    payload = UsdGeom.Xform.Define(stage, "/World/DeliveryPayload")
    tray_cfg = config["payload"]["tray"]
    tray = UsdGeom.Cube.Define(stage, "/World/DeliveryPayload/Tray")
    tray.CreateSizeAttr(1.0)
    tray.CreateDisplayColorAttr([Gf.Vec3f(*tray_cfg["color_rgb"])])
    tray_xform = UsdGeom.XformCommonAPI(tray)
    tray_xform.SetScale(Gf.Vec3f(*tray_cfg["size_xyz_m"]))

    cup_cfg = config["payload"]["cup"]
    cup = UsdGeom.Cylinder.Define(stage, "/World/DeliveryPayload/CoffeeCup")
    cup.CreateAxisAttr("Z")
    cup.CreateRadiusAttr(float(cup_cfg["radius_m"]))
    cup.CreateHeightAttr(float(cup_cfg["height_m"]))
    cup.CreateDisplayColorAttr([Gf.Vec3f(*cup_cfg["color_rgb"])])

    goal = UsdGeom.Cylinder.Define(stage, "/World/Navigation/Target")
    goal.CreateAxisAttr("Z")
    goal.CreateRadiusAttr(float(config["success"]["target_radius_m"]))
    goal.CreateHeightAttr(0.025)
    goal.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.95, 0.35)])
    UsdGeom.XformCommonAPI(goal).SetTranslate(
        Gf.Vec3d(float(target[0]), float(target[1]), 0.015)
    )
    if include_route:
        stride = max(1, len(route.waypoints) // 100)
        for index in range(0, len(route.waypoints), stride):
            x, y = route.waypoints[index]
            marker = UsdGeom.Sphere.Define(
                stage, f"/World/Navigation/Route/P_{index:04d}"
            )
            marker.CreateRadiusAttr(0.06)
            marker.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.55, 0.06)])
            UsdGeom.XformCommonAPI(marker).SetTranslate(Gf.Vec3d(x, y, 0.04))

    # Neutral visual floor: the reconstructed fused collider frequently has no
    # readable material, while the learned route still needs visual contrast.
    points = list(route.waypoints) + [route.target_xy]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    floor_margin = 1.4
    floor = UsdGeom.Mesh.Define(stage, "/World/Navigation/DemoFloor")
    floor.CreatePointsAttr(
        [
            Gf.Vec3f(min(xs) - floor_margin, min(ys) - floor_margin, 0.003),
            Gf.Vec3f(max(xs) + floor_margin, min(ys) - floor_margin, 0.003),
            Gf.Vec3f(max(xs) + floor_margin, max(ys) + floor_margin, 0.003),
            Gf.Vec3f(min(xs) - floor_margin, max(ys) + floor_margin, 0.003),
        ]
    )
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateSubdivisionSchemeAttr("none")
    floor.CreateDoubleSidedAttr(True)
    floor.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.33, 0.37)])
    video_cfg = config.get("video", {})
    if video_cfg.get("hide_floor_visual", False):
        # In the pano-origin view the panorama itself supplies the café floor;
        # any synthetic floor plane would paint over it.
        UsdGeom.Imageable(floor).MakeInvisible()

    light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    # A textured dome carries the panorama's own brightness; the video config can
    # dial the lights down so the white Carter body is not blown out on the floor.
    light.CreateIntensityAttr(float(video_cfg.get("dome_intensity", 1800.0)))
    light.CreateExposureAttr(float(video_cfg.get("dome_exposure", 2.0)))
    panorama_value = config["environment"].get("panorama")
    if panorama_value:
        panorama = resolve_path(panorama_value)
        if panorama.is_file():
            light.CreateTextureFileAttr(Sdf.AssetPath(str(panorama)))
            light.CreateTextureFormatAttr("latlong")
    dome_yaw = float(video_cfg.get("dome_yaw_deg", 0.0))
    if dome_yaw:
        # Rotate the panorama about Z so its floor layout lines up with the metric
        # route; tune video.dome_yaw_deg in the config without touching code.
        UsdGeom.XformCommonAPI(light).SetRotate(Gf.Vec3f(0.0, 0.0, dome_yaw))
    key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    key.CreateIntensityAttr(float(video_cfg.get("key_intensity", 5000.0)))
    key.CreateExposureAttr(float(video_cfg.get("key_exposure", 1.5)))
    UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(-35.0, 25.0, 20.0))

    fill = UsdLux.SphereLight.Define(stage, "/World/FillLight")
    fill.CreateIntensityAttr(float(video_cfg.get("fill_intensity", 35000.0)))
    fill.CreateRadiusAttr(2.0)
    UsdGeom.XformCommonAPI(fill).SetTranslate(
        Gf.Vec3d(0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys)), 5.0)
    )
    return tray, cup


def sync_payload(
    tray_prim: Any,
    cup_prim: Any,
    position: Sequence[float],
    yaw: float,
    config: dict[str, Any],
) -> None:
    from pxr import Gf, UsdGeom

    tray_position, _ = rotated_offset_pose(
        position, yaw, config["payload"]["tray"]["offset_xyz_m"]
    )
    cup_position, _ = rotated_offset_pose(
        position, yaw, config["payload"]["cup"]["offset_xyz_m"]
    )
    rotation = Gf.Vec3f(0.0, 0.0, math.degrees(yaw))
    tray_xform = UsdGeom.XformCommonAPI(tray_prim)
    tray_xform.SetTranslate(Gf.Vec3d(*tray_position))
    tray_xform.SetRotate(rotation)
    cup_xform = UsdGeom.XformCommonAPI(cup_prim)
    cup_xform.SetTranslate(Gf.Vec3d(*cup_position))
    cup_xform.SetRotate(rotation)


class IsaacVideoRecorder:
    """Small RGB annotator -> OpenCV MP4 bridge for the demo episode."""

    def __init__(
        self,
        *,
        path: Path,
        route: RouteData,
        config: dict[str, Any],
    ) -> None:
        import cv2
        import omni.replicator.core as rep

        self.cv2 = cv2
        self.rep = rep
        video_cfg = config.get("video", {})
        self.video_cfg = dict(video_cfg)
        self.width = int(video_cfg.get("width", 960))
        self.height = int(video_cfg.get("height", 540))
        self.playback_fps = float(video_cfg.get("playback_fps", 12.0))
        self.capture_hz = float(video_cfg.get("capture_hz", 5.0))
        self.camera_mode = str(video_cfg.get("camera_mode") or "overview")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Video width and height must be positive")
        if self.playback_fps <= 0 or self.capture_hz <= 0:
            raise ValueError("Video playback_fps and capture_hz must be positive")

        self.path = path if path.suffix else path.with_suffix(".mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        points = list(route.waypoints) + [route.target_xy]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        margin = float(video_cfg.get("route_margin_m", 1.0))
        min_x, max_x = min(xs) - margin, max(xs) + margin
        min_y, max_y = min(ys) - margin, max(ys) + margin
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        span = max(max_x - min_x, max_y - min_y)
        camera_height = max(
            float(video_cfg.get("camera_min_height_m", 9.0)),
            span * float(video_cfg.get("camera_height_scale", 1.85)),
        )

        # A shallow oblique offset avoids the mathematically singular
        # straight-down look-at pose while retaining an overhead overview.
        camera_position = (
            center_x + 0.10 * span,
            center_y - 0.18 * span,
            camera_height,
        )
        camera_look_at = (center_x, center_y, 0.1)
        if self.camera_mode == "pano_origin":
            # The World Labs panorama on the dome light was captured at the Marble
            # frame origin, which the grid transform maps to (0, 0, ground_offset).
            # A camera at exactly that point sees the dome with zero parallax, so the
            # robot driving on the metric floor reads as being inside the café.
            camera_position = tuple(
                float(v) for v in video_cfg.get("camera_position_m", (0.0, 0.0, 1.6712))
            )
            camera_look_at = (
                center_x,
                center_y,
                float(video_cfg.get("look_at_z_m", 0.3)),
            )
        elif self.camera_mode == "follow":
            camera_position, camera_look_at = self._follow_pose(
                (route.waypoints[0][0], route.waypoints[0][1], 0.0),
                initial_yaw(route, 0.0),
                video_cfg,
            )
        self.camera_position = camera_position
        self.camera_look_at = camera_look_at

        rep.orchestrator.set_capture_on_play(False)
        self.camera = rep.functional.create.camera(
            position=camera_position,
            look_at=camera_look_at,
            focal_length=float(video_cfg.get("focal_length_mm", 18.0)),
            clipping_range=(0.05, max(100.0, camera_height * 4.0)),
            parent="/World",
            name="World2WorkDemoCamera",
        )
        self.render_product = rep.create.render_product(
            self.camera,
            (self.width, self.height),
            name="World2WorkDemoRenderProduct",
        )
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach(self.render_product)

        codec_candidates = ["mp4v", "avc1"]
        self.writer = None
        for codec in codec_candidates:
            candidate = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*codec),
                self.playback_fps,
                (self.width, self.height),
            )
            if candidate.isOpened():
                self.writer = candidate
                self.codec = codec
                break
            candidate.release()
        if self.writer is None:
            raise RuntimeError(
                f"OpenCV could not open an MP4 writer for {self.path}"
            )
        self.frame_count = 0

    @staticmethod
    def _follow_pose(
        position: Sequence[float],
        yaw: float,
        video_cfg: dict[str, Any],
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        distance = float(video_cfg.get("follow_distance_m", 2.6))
        lateral = float(video_cfg.get("follow_lateral_m", 0.75))
        height = float(video_cfg.get("follow_height_m", 1.75))
        look_ahead = float(video_cfg.get("follow_look_ahead_m", 0.45))
        look_height = float(video_cfg.get("follow_look_at_height_m", 0.48))
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        px, py, pz = (float(value) for value in position[:3])
        return (
            (
                px - distance * cos_yaw - lateral * sin_yaw,
                py - distance * sin_yaw + lateral * cos_yaw,
                pz + height,
            ),
            (
                px + look_ahead * cos_yaw,
                py + look_ahead * sin_yaw,
                pz + look_height,
            ),
        )

    def capture(
        self,
        position: Sequence[float] | None = None,
        yaw: float | None = None,
    ) -> bool:
        import numpy as np

        if self.camera_mode == "follow" and position is not None and yaw is not None:
            camera_position, look_at = self._follow_pose(
                position, yaw, self.video_cfg
            )
            self.rep.functional.modify.pose(
                self.camera,
                position_value=camera_position,
                look_at_value=look_at,
                look_at_up_axis=(0, 0, 1),
                write_to_usd=True,
            )
        # Isaac Sim 6 does not populate attached annotators from
        # World.step(render=True) when capture-on-play is disabled.  Trigger a
        # zero-delta Replicator capture explicitly so physics time is unchanged.
        self.rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
        payload = self.annotator.get_data()
        if isinstance(payload, dict):
            payload = payload.get("data")
        if payload is None:
            return False
        rgb = np.asarray(payload)
        if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
            return False
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
            if maximum <= 1.0:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = self.cv2.resize(rgb, (self.width, self.height))
        bgr = self.cv2.cvtColor(np.ascontiguousarray(rgb), self.cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)
        self.frame_count += 1
        return True

    def release_writer(self) -> None:
        # Finalize the MP4 (moov atom) as early as possible: a native crash in
        # Replicator teardown must not cost the frames already encoded.
        if getattr(self, "writer", None) is not None:
            try:
                self.writer.release()
            finally:
                self.writer = None

    def close(self) -> None:
        self.release_writer()
        try:
            self.rep.orchestrator.wait_until_complete()
        except Exception:
            pass
        try:
            self.annotator.detach()
        except Exception:
            pass
        try:
            self.render_product.destroy()
        except Exception:
            pass
        video_valid = (
            self.frame_count > 0
            and self.path.is_file()
            and self.path.stat().st_size > 256
        )
        print(
            ("VIDEO_OK " if video_valid else "VIDEO_FAILED ")
            + json.dumps(
                {
                    "output": str(self.path),
                    "frames": self.frame_count,
                    "capture_hz": self.capture_hz,
                    "playback_fps": self.playback_fps,
                    "resolution": [self.width, self.height],
                    "codec": self.codec,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def run_isaac(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    route: RouteData,
    target_id: str,
    grid_meta: dict[str, Any],
    writer: JsonlWriter,
    summary_path: Path,
    episode_id: str,
) -> dict[str, Any]:
    from isaacsim import SimulationApp

    video_enabled = args.video_output is not None
    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
            "sync_loads": True,
            "disable_viewport_updates": not video_enabled,
        }
    )
    start_record: dict[str, Any] | None = None
    follower = RouteFollower(route, config)
    elapsed = 0.0
    total_reward = 0.0
    actual_distance = 0.0
    previous_position_xy: Any = None
    # With the default flat-ground execution there are no obstacle colliders,
    # so obstacle collisions are exactly zero.  When Marble/proxy collision is
    # enabled, report null until a contact sensor is explicitly attached rather
    # than fabricating a count from wheel slip.
    collision_count: int | None = (
        None if args.enable_marble_collision or args.enable_proxies else 0
    )
    last_error = float("inf")
    video_recorder: IsaacVideoRecorder | None = None
    # Rendering through Replicator can change the Kit timeline state in Isaac
    # Sim 6.  Keep the control rollout render-free and retain only poses from
    # the completed physics trajectory for a separate, post-episode replay.
    video_pose_samples: list[tuple[list[float], list[float]]] = []
    try:
        import numpy as np
        from isaacsim.core.api import World
        # Isaac Sim 6 exposes the single-robot wrapper from core.prims. The
        # legacy core.api.articulations package only contains helper classes.
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.experimental.utils.app import enable_extension
        from isaacsim.storage.native import get_assets_root_path
        from pxr import UsdPhysics

        enable_extension("omni.kit.asset_converter")
        for _ in range(3):
            simulation_app.update()
        world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / int(config["logging"]["physics_hz"]),
            rendering_dt=1.0 / int(config["logging"]["state_hz"]),
        )
        world.scene.add_default_ground_plane()
        stage = stage_utils.get_current_stage()
        if args.video_output is not None and config.get("video", {}).get(
            "hide_floor_visual", False
        ):
            # Keep the ground plane for physics but hide its grid so the panorama
            # floor is visible from the pano-origin camera. Visibility does not
            # affect PhysX collision.
            from pxr import UsdGeom as _UsdGeom

            ground_prim = stage.GetPrimAtPath("/World/defaultGroundPlane")
            if ground_prim and ground_prim.IsValid():
                _UsdGeom.Imageable(ground_prim).MakeInvisible()
                print("GROUND_VISUAL_HIDDEN /World/defaultGroundPlane", flush=True)
        scale, ground_offset, visual_bias = environment_transform(config, grid_meta)

        marble_meshes = 0
        converted: Path | None = None
        marble_source, _, _ = marble_source_paths(config, args.marble_source)
        marble_summary: dict[str, Any] | None = None
        if not args.disable_marble_visual:
            converted, marble_meshes, marble_summary = add_marble_scene(
                simulation_app=simulation_app,
                stage=stage,
                config=config,
                scale=scale,
                ground_offset=ground_offset,
                visual_bias=visual_bias,
                enable_collision=args.enable_marble_collision,
                marble_source=args.marble_source,
            )
        if not args.disable_marble_visual and config.get("video", {}).get("marble_matte"):
            # Occlusion fix for the pano-origin shot: keep the fused Marble mesh in
            # the render as an RTX *matte object*. It draws the dome background
            # instead of its own surface but still occludes the robot, so the
            # Carter disappears behind café tables/chairs instead of floating
            # over them.
            try:
                import carb
                from pxr import Sdf as _Sdf, UsdGeom as _UsdGeom

                carb.settings.get_settings().set("/rtx/matteObject/enabled", True)
                matte_count = 0
                for prim in stage.Traverse():
                    if str(prim.GetPath()).startswith("/World/MarbleCafe") and prim.IsA(_UsdGeom.Mesh):
                        _UsdGeom.Imageable(prim).MakeVisible()
                        _UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
                            "isMatteObject", _Sdf.ValueTypeNames.Bool
                        ).Set(True)
                        matte_count += 1
                print(f"MARBLE_MATTE_OCCLUDER meshes={matte_count}", flush=True)
            except Exception as matte_exc:
                print(f"MARBLE_MATTE_WARN {matte_exc}", flush=True)
        proxy_count = add_proxy_obstacles(world, grid_meta) if args.enable_proxies else 0

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac Sim asset root is unavailable")
        robot_asset = resolve_robot_asset(assets_root, config, args.robot_usd)
        robot_prim_path = "/World/NovaCarter"
        robot_variants = None
        if "novacarter" in robot_asset.lower():
            robot_variants = [
                ("Physics", "physx"),
                ("Sensors", "None"),
                ("Configuration", "Base"),
                ("ROS", "Disabled"),
            ]
        if robot_variants is None:
            stage_utils.add_reference_to_stage(robot_asset, robot_prim_path)
        else:
            stage_utils.add_reference_to_stage(
                robot_asset, robot_prim_path, variants=robot_variants
            )
        articulation_roots: list[str] = []
        prim_count = 0
        for _ in range(480):
            simulation_app.update()
            prim_count = sum(
                1
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(robot_prim_path)
            )
            articulation_roots = [
                str(prim.GetPath())
                for prim in stage.Traverse()
                if str(prim.GetPath()).startswith(robot_prim_path)
                and prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            ]
            if prim_count >= 5 and articulation_roots:
                break
        if prim_count < 5:
            raise RuntimeError(f"Nova Carter reference did not resolve: {robot_asset}")

        if not articulation_roots:
            raise RuntimeError(
                "Nova Carter asset contains no articulation root under "
                f"{robot_prim_path}"
            )
        articulation_prim_path = next(
            (
                path
                for path in articulation_roots
                if path.lower().endswith("/chassis_link")
            ),
            articulation_roots[0],
        )

        robot = world.scene.add(
            SingleArticulation(prim_path=articulation_prim_path, name="nova_carter")
        )
        start_x, start_y = route.waypoints[0]
        start_z = float(config["robot"]["start_z_m"])
        start_yaw = initial_yaw(
            route,
            float(config["navigation"].get("fallback_start_yaw_rad", 0.0)),
        )
        start_position = np.array([start_x, start_y, start_z], dtype=np.float64)
        start_orientation = np.array(yaw_to_wxyz(start_yaw), dtype=np.float64)
        robot.set_default_state(
            position=start_position, orientation=start_orientation
        )

        target = target_xyz(config, route, target_id)
        tray_prim, cup_prim = add_visuals(
            stage, route, config, target, not args.no_route_markers
        )
        world.reset()
        robot.set_world_pose(start_position, start_orientation)
        world.step(render=False)
        dof_names = list(robot.dof_names or [])
        print(f"CARTER_DOF_NAMES {json.dumps(dof_names)}", flush=True)
        left_name, right_name, left_index, right_index = discover_wheel_dofs(
            dof_names, config["robot"]["wheel_dof_name_pairs"]
        )

        wheel_radius = float(config["robot"]["wheel_radius_m"])
        wheel_base = float(config["robot"]["wheel_base_m"])
        if "carter_v1" in robot_asset.lower() and "novacarter" not in robot_asset.lower():
            wheel_radius, wheel_base = 0.24, 0.54
            print(
                "CARTER_ASSET_FALLBACK using carter_v1 geometry "
                "wheel_radius=0.24 wheel_base=0.54",
                flush=True,
            )
        signs = config["robot"].get("wheel_velocity_signs", [1.0, 1.0])
        start_record = build_start_record(
            episode_id=episode_id,
            config=config,
            route=route,
            target_id=target_id,
            scale=scale,
            ground_offset=ground_offset,
            marble_collision=args.enable_marble_collision,
            proxies=args.enable_proxies,
            dry_run=False,
            asset_usd=robot_asset,
            wheel_names=[left_name, right_name],
            wheel_radius=wheel_radius,
            wheel_base=wheel_base,
            semantic_scene=load_semantic_scene(config, target_id, grid_meta),
            marble_source=marble_source,
        )
        writer.write(start_record)
        print(
            "ISAAC_NAV_READY "
            + json.dumps(
                {
                    "asset": robot_asset,
                    "articulation_root": articulation_prim_path,
                    "wheel_dofs": [left_name, right_name],
                    "marble_meshes": marble_meshes,
                    "marble_source": marble_source,
                    "marble_scene": marble_summary,
                    "marble_usd": str(converted) if converted else None,
                    "proxy_count": proxy_count,
                    "route_waypoints": len(route.waypoints),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        physics_hz = int(config["logging"]["physics_hz"])
        state_hz = int(config["logging"]["state_hz"])
        if physics_hz <= 0 or state_hz <= 0 or physics_hz % state_hz:
            raise ValueError("physics_hz must be a positive multiple of state_hz")
        log_interval = physics_hz // state_hz
        capture_hz = float(config.get("video", {}).get("capture_hz", 5.0))
        if video_enabled and capture_hz <= 0:
            raise ValueError("video.capture_hz must be positive")
        video_interval = (
            max(1, round(physics_hz / capture_hz))
            if video_enabled
            else 0
        )
        dt = 1.0 / physics_hz
        timeout = float(args.timeout_s or config["success"]["timeout_s"])
        previous_error = math.hypot(target[0] - start_x, target[1] - start_y)
        shortest_distance = shortest_distance_m(route, grid_meta, target_id)
        last_progress_position = np.array([start_x, start_y], dtype=np.float64)
        last_progress_time = 0.0
        cup_offset = config["payload"]["cup"]["offset_xyz_m"]
        outcome: dict[str, Any] | None = None

        if video_enabled:
            initial_position, initial_orientation = robot.get_world_pose()
            video_pose_samples.append(
                (
                    np.asarray(initial_position, dtype=np.float64).tolist(),
                    np.asarray(initial_orientation, dtype=np.float64).tolist(),
                )
            )

        for frame in range(int(math.ceil(timeout * physics_hz)) + 1):
            elapsed = frame * dt
            position, orientation = robot.get_world_pose()
            position = np.asarray(position, dtype=np.float64)
            orientation = np.asarray(orientation, dtype=np.float64)
            yaw = wxyz_to_yaw(orientation)
            linear, angular, phase, reached = follower.command(
                float(position[0]), float(position[1]), yaw, elapsed
            )
            stalled = False
            if linear > 0.05:
                moved = float(np.linalg.norm(position[:2] - last_progress_position))
                if moved >= float(config["success"]["stall_min_progress_m"]):
                    last_progress_position = position[:2].copy()
                    last_progress_time = elapsed
                elif elapsed - last_progress_time >= float(
                    config["success"]["stall_timeout_s"]
                ):
                    stalled = True
                    linear, angular, phase = 0.0, 0.0, "STALLED"
            else:
                last_progress_position = position[:2].copy()
                last_progress_time = elapsed

            timed_out = elapsed >= timeout and not reached
            if timed_out:
                linear, angular, phase = 0.0, 0.0, "TIMEOUT"
            command_wheels = wheel_speeds(
                linear, angular, wheel_radius, wheel_base, signs
            )
            action = ArticulationAction(
                joint_velocities=np.array(command_wheels, dtype=np.float32),
                joint_indices=np.array([left_index, right_index], dtype=np.int32),
            )
            robot.apply_action(action)
            sync_payload(tray_prim, cup_prim, position, yaw, config)
            # The physics loop never invokes Replicator or renders.  Video is
            # generated after the outcome from sampled, actual simulator poses.
            world.step(render=False)

            new_position, new_orientation = robot.get_world_pose()
            new_position = np.asarray(new_position, dtype=np.float64)
            new_orientation = np.asarray(new_orientation, dtype=np.float64)
            if previous_position_xy is None:
                previous_position_xy = np.array([start_x, start_y], dtype=np.float64)
            actual_distance += float(
                np.linalg.norm(new_position[:2] - previous_position_xy)
            )
            previous_position_xy = new_position[:2].copy()
            linear_velocity = np.asarray(robot.get_linear_velocity(), dtype=np.float64)
            angular_velocity = np.asarray(robot.get_angular_velocity(), dtype=np.float64)
            joint_velocity = np.asarray(robot.get_joint_velocities(), dtype=np.float64)
            actual_wheels = [joint_velocity[left_index], joint_velocity[right_index]]
            error = math.hypot(target[0] - new_position[0], target[1] - new_position[1])
            reward = (
                (previous_error - error) * float(config["reward"]["progress_scale"])
                + float(config["reward"]["step_penalty"])
            )
            done = reached or timed_out or stalled
            if reached:
                reward += float(config["reward"]["success_bonus"])
            elif timed_out:
                reward += float(config["reward"]["timeout_penalty"])
            elif stalled:
                reward += float(config["reward"]["stall_penalty"])
            total_reward += reward
            previous_error = error
            last_error = error

            if video_enabled and ((frame + 1) % video_interval == 0 or done):
                pose_sample = (new_position.tolist(), new_orientation.tolist())
                if not video_pose_samples or pose_sample != video_pose_samples[-1]:
                    video_pose_samples.append(pose_sample)

            if frame % log_interval == 0 or done:
                writer.write(
                    build_step_record(
                        episode_id=episode_id,
                        step_index=writer.logged_steps,
                        elapsed_s=elapsed,
                        phase=phase,
                        position=new_position,
                        orientation_wxyz=new_orientation,
                        linear_velocity=linear_velocity,
                        angular_velocity=angular_velocity,
                        wheel_velocity=actual_wheels,
                        command=(linear, angular),
                        cup_offset=cup_offset,
                        target_id=target_id,
                        target_xyz=target,
                        policy_cell=follower.policy_cell(),
                        waypoint_index=follower.waypoint_index,
                        reward=reward,
                        done=done,
                    )
                )
            if done:
                success = reached
                reason = (
                    "target_reached" if reached else "stalled" if stalled else "timeout"
                )
                outcome = build_outcome(
                    episode_id=episode_id,
                    success=success,
                    reason=reason,
                    elapsed_s=elapsed,
                    xy_error=error,
                    writer=writer,
                    follower=follower,
                    total_reward=total_reward,
                    dry_run=False,
                    actual_distance_m=actual_distance,
                    shortest_distance=shortest_distance,
                    collisions=collision_count,
                )
                break

        if outcome is None:
            raise RuntimeError("Isaac simulation loop ended without an outcome")
        writer.write(outcome)
        write_summary(summary_path, start_record, outcome)
        physics_marker = "PHYSICS_EPISODE_OK" if outcome["success"] else "PHYSICS_EPISODE_FAILED"
        print(f"{physics_marker} {json.dumps(outcome, sort_keys=True)}", flush=True)

        if video_enabled:
            # The navigation metrics above are final before any renderer is
            # initialized.  Pause the timeline, then move the robot through the
            # sampled physics poses solely for visual replay.  Any video failure
            # is non-fatal and cannot rewrite a valid navigation outcome.
            try:
                # Keep the timeline playing: with it paused, set_world_pose() on
                # the articulation never reaches the renderer (Fabric keeps the
                # last simulated pose), so the robot body froze while the tray
                # moved. A zero-velocity physics step per replayed pose flushes
                # the transform into the render scene. Metrics are already final.
                video_recorder = IsaacVideoRecorder(
                    path=args.video_output,
                    route=route,
                    config=config,
                )
                print(
                    "VIDEO_REPLAY_BEGIN "
                    + json.dumps(
                        {
                            "samples": len(video_pose_samples),
                            "capture_hz": capture_hz,
                            "provenance": "actual_isaac_physics_pose_replay",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                # The simulated articulation's transforms live in Fabric and do
                # not reach the renderer once we drive it by hand (the body froze
                # while the USD-authored tray moved). So stop physics, hide the
                # simulated robot, and replay the *recorded physics poses* on a
                # physics-free visual clone of the same Nova Carter asset that is
                # moved through plain USD xforms -- the path that demonstrably
                # renders.
                import omni.timeline
                from pxr import Gf as _Gf, UsdGeom as _UsdGeom

                omni.timeline.get_timeline_interface().stop()
                for _ in range(3):
                    simulation_app.update()
                sim_robot_prim = stage.GetPrimAtPath(robot_prim_path)
                if sim_robot_prim and sim_robot_prim.IsValid():
                    _UsdGeom.Imageable(sim_robot_prim).MakeInvisible()
                # Author all replay xform ops on a fresh parent Xform: the asset's
                # own root already carries xformOp:scale with a different precision,
                # and AddScaleOp on it raises inside USD.
                clone_path = "/World/CarterVisualRoot"
                _UsdGeom.Xform.Define(stage, clone_path)
                stage_utils.add_reference_to_stage(robot_asset, clone_path + "/CarterVisual")
                for _ in range(30):
                    simulation_app.update()
                clone_prim = stage.GetPrimAtPath(clone_path)
                # Own the clone's xform op stack outright so per-frame writes
                # cannot be rejected by ops the asset already authored.
                clone_xformable = _UsdGeom.Xformable(clone_prim)
                clone_xformable.ClearXformOpOrder()
                clone_translate_op = clone_xformable.AddTranslateOp()
                clone_rotate_op = clone_xformable.AddRotateZOp()
                # Visual-only presentation knobs for the demo shot.
                _vcfg = config.get("video", {})
                robot_visual_scale = float(_vcfg.get("robot_visual_scale", 1.0))
                clone_xformable.AddScaleOp().Set(
                    _Gf.Vec3f(robot_visual_scale, robot_visual_scale, robot_visual_scale)
                )
                replay_config = config
                if robot_visual_scale != 1.0:
                    import copy as _copy

                    replay_config = _copy.deepcopy(config)
                    for key in ("tray", "cup"):
                        replay_config["payload"][key]["offset_xyz_m"] = [
                            float(v) * robot_visual_scale
                            for v in config["payload"][key]["offset_xyz_m"]
                        ]
                    for prim_obj in (tray_prim, cup_prim):
                        try:
                            _UsdGeom.XformCommonAPI(prim_obj).SetScale(
                                _Gf.Vec3f(robot_visual_scale, robot_visual_scale, robot_visual_scale)
                            )
                        except Exception as scale_exc:
                            print(f"PAYLOAD_SCALE_WARN {scale_exc}", flush=True)
                if _vcfg.get("table_prop"):
                    # A simple café table at the delivery target so "table 7" is an object.
                    table_scale = float(_vcfg.get("table_prop_scale", robot_visual_scale))
                    table_positions = [(float(route.target_xy[0]), float(route.target_xy[1]))]
                    table_positions += [(float(p[0]), float(p[1])) for p in _vcfg.get("table_prop_extra_xy", [])]
                    for t_index, (tx, ty) in enumerate(table_positions):
                        root = f"/World/TableProp{t_index}"
                        top = _UsdGeom.Cylinder.Define(stage, root + "/Top")
                        top.CreateRadiusAttr(0.45 * table_scale)
                        top.CreateHeightAttr(0.04 * table_scale)
                        top.CreateAxisAttr("Z")
                        top.CreateDisplayColorAttr([_Gf.Vec3f(0.55, 0.33, 0.16)])
                        _UsdGeom.XformCommonAPI(top).SetTranslate(_Gf.Vec3d(tx, ty, 0.72 * table_scale))
                        leg = _UsdGeom.Cylinder.Define(stage, root + "/Leg")
                        leg.CreateRadiusAttr(0.04 * table_scale)
                        leg.CreateHeightAttr(0.70 * table_scale)
                        leg.CreateAxisAttr("Z")
                        leg.CreateDisplayColorAttr([_Gf.Vec3f(0.2, 0.2, 0.22)])
                        _UsdGeom.XformCommonAPI(leg).SetTranslate(_Gf.Vec3d(tx, ty, 0.35 * table_scale))
                        base = _UsdGeom.Cylinder.Define(stage, root + "/Base")
                        base.CreateRadiusAttr(0.25 * table_scale)
                        base.CreateHeightAttr(0.02 * table_scale)
                        base.CreateAxisAttr("Z")
                        base.CreateDisplayColorAttr([_Gf.Vec3f(0.2, 0.2, 0.22)])
                        _UsdGeom.XformCommonAPI(base).SetTranslate(_Gf.Vec3d(tx, ty, 0.01 * table_scale))
                    print(f"TABLE_PROP_ADDED count={len(table_positions)}", flush=True)
                print(
                    "VIDEO_REPLAY_CLONE "
                    + json.dumps({"clone": clone_path, "valid": bool(clone_prim and clone_prim.IsValid())}),
                    flush=True,
                )
                # Optional visual-only correction if the panorama's implied scale
                # disagrees with the Marble metric estimate: scales the replayed
                # XY about the camera so the robot sits where the pano floor is.
                pose_scale = float(config.get("video", {}).get("pose_scale", 1.0))
                for replay_position, replay_orientation in video_pose_samples:
                    replay_position_array = np.asarray(
                        replay_position, dtype=np.float64
                    )
                    if pose_scale != 1.0:
                        replay_position_array = replay_position_array.copy()
                        replay_position_array[:2] *= pose_scale
                    replay_orientation_array = np.asarray(
                        replay_orientation, dtype=np.float64
                    )
                    replay_yaw = wxyz_to_yaw(replay_orientation_array)
                    clone_translate_op.Set(
                        _Gf.Vec3d(
                            float(replay_position_array[0]),
                            float(replay_position_array[1]),
                            float(replay_position_array[2]) * robot_visual_scale,
                        )
                    )
                    clone_rotate_op.Set(float(math.degrees(replay_yaw)))
                    sync_payload(
                        tray_prim,
                        cup_prim,
                        replay_position_array,
                        replay_yaw,
                        replay_config,
                    )
                    simulation_app.update()
                    if video_recorder.frame_count + 1 >= len(video_pose_samples):
                        pass
                    video_recorder.capture(
                        position=replay_position_array,
                        yaw=replay_yaw,
                    )
                video_recorder.release_writer()
                print(
                    "VIDEO_FRAMES_FLUSHED "
                    + json.dumps({"frames": video_recorder.frame_count}),
                    flush=True,
                )
            except Exception as video_exc:
                print(
                    "VIDEO_REPLAY_FAILED "
                    + json.dumps(
                        {
                            "error": f"{type(video_exc).__name__}: {video_exc}",
                            "provenance": "actual_isaac_physics_pose_replay",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            finally:
                if video_recorder is not None:
                    try:
                        video_recorder.close()
                    except Exception as close_exc:
                        print(
                            "VIDEO_REPLAY_FAILED "
                            + json.dumps(
                                {
                                    "error": (
                                        f"{type(close_exc).__name__}: {close_exc}"
                                    ),
                                    "phase": "close",
                                    "provenance": (
                                        "actual_isaac_physics_pose_replay"
                                    ),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    finally:
                        video_recorder = None
        return outcome
    except BaseException as exc:
        traceback.print_exc()
        if start_record is not None:
            failure = build_outcome(
                episode_id=episode_id,
                success=False,
                reason="simulation_error",
                elapsed_s=elapsed,
                xy_error=last_error if math.isfinite(last_error) else 1e9,
                writer=writer,
                follower=follower,
                total_reward=total_reward,
                dry_run=False,
                actual_distance_m=actual_distance,
                shortest_distance=route.path_length_m,
                collisions=collision_count,
            )
            writer.write(failure)
            write_summary(summary_path, start_record, failure)
        raise exc
    finally:
        if video_recorder is not None:
            video_recorder.close()
        simulation_app.close()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    target_id = args.target or str(config["target_id"])
    nav = config["navigation"]
    route_path = args.route or resolve_path(
        nav["route_dir"]
    ) / str(nav["route_filename_pattern"]).format(target_id=target_id)
    grid_meta_path = args.grid_meta or resolve_path(nav["grid_meta"])
    grid_meta = load_json(grid_meta_path) if grid_meta_path.is_file() else {}
    route = slice_route(
        load_route(route_path, grid_meta, config, target_id), args.start_index
    )
    episode_id = args.episode_id or (
        f"{target_id}-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
    )
    jsonl_path, summary_path = output_paths(args, episode_id)
    writer = JsonlWriter(jsonl_path)
    try:
        if args.dry_run:
            outcome = run_dry(
                args=args,
                config=config,
                route=route,
                target_id=target_id,
                grid_meta=grid_meta,
                writer=writer,
                summary_path=summary_path,
                episode_id=episode_id,
            )
        else:
            outcome = run_isaac(
                args=args,
                config=config,
                route=route,
                target_id=target_id,
                grid_meta=grid_meta,
                writer=writer,
                summary_path=summary_path,
                episode_id=episode_id,
            )
    finally:
        writer.close()
    marker = "EPISODE_OK" if outcome["success"] else "EPISODE_FAILED"
    print(f"{marker} {json.dumps(outcome, sort_keys=True)}", flush=True)
    if not outcome["success"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
