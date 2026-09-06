#!/usr/bin/env python3
"""Build a small navigation occupancy grid from a World Labs Marble collider.

This script deliberately has no third-party dependencies.  It reads the GLB
directly, applies the verified Marble Y-up -> Isaac Z-up metric transform,
rasterizes floor support and low obstacles, keeps the largest connected
walkable island, and writes JSON + PNG artifacts for the navigation demo.

The collider is a fused, unlabeled reconstruction.  Table semantics come from
scripts/segment_marble_collider.py (run automatically when its output is
missing or stale): every detected table top gets an *approach cell*, the
nearest reachable free cell next to its footprint, and the demo target id
(table_7 by default) is aliased to one of those physical tables.  When no
semantics are available the old behaviour remains: table_7 is a reachable
free-space goal proxy, and grid_meta.json says so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import zlib
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence


COMPONENT_FORMATS = {
    5120: ("b", 1),
    5121: ("B", 1),
    5122: ("h", 2),
    5123: ("H", 2),
    5125: ("I", 4),
    5126: ("f", 4),
}
TYPE_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def _mat_identity() -> list[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _mat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    out = [0.0] * 16
    for col in range(4):
        for row in range(4):
            out[col * 4 + row] = sum(a[k * 4 + row] * b[col * 4 + k] for k in range(4))
    return out


def _node_matrix(node: dict) -> list[float]:
    if "matrix" in node:
        return [float(v) for v in node["matrix"]]
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    sx, sy, sz = node.get("scale", [1, 1, 1])
    tx, ty, tz = node.get("translation", [0, 0, 0])
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    return [
        (1 - (yy + zz)) * sx, (xy + wz) * sx, (xz - wy) * sx, 0,
        (xy - wz) * sy, (1 - (xx + zz)) * sy, (yz + wx) * sy, 0,
        (xz + wy) * sz, (yz - wx) * sz, (1 - (xx + yy)) * sz, 0,
        tx, ty, tz, 1,
    ]


def _transform_point(m: Sequence[float], p: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = p
    return (
        m[0] * x + m[4] * y + m[8] * z + m[12],
        m[1] * x + m[5] * y + m[9] * z + m[13],
        m[2] * x + m[6] * y + m[10] * z + m[14],
    )


def _read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"{path} is not a binary glTF (GLB) file")
    version, declared = struct.unpack_from("<II", data, 4)
    if version != 2 or declared > len(data):
        raise ValueError(f"unsupported/corrupt GLB: version={version}, declared={declared}, bytes={len(data)}")
    cursor = 12
    gltf = None
    binary = None
    while cursor + 8 <= declared:
        length, kind = struct.unpack_from("<II", data, cursor)
        chunk = data[cursor + 8: cursor + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk.rstrip(b"\x00 \t\r\n").decode("utf-8"))
        elif kind == 0x004E4942:
            binary = bytes(chunk)
        cursor += 8 + length
    if gltf is None or binary is None:
        raise ValueError("GLB is missing its JSON or BIN chunk")
    return gltf, binary


def _decode_accessor(gltf: dict, binary: bytes, index: int) -> list:
    accessor = gltf["accessors"][index]
    if "sparse" in accessor:
        raise ValueError(f"sparse accessor {index} is not supported")
    view = gltf["bufferViews"][accessor["bufferView"]]
    fmt, component_bytes = COMPONENT_FORMATS[accessor["componentType"]]
    width = TYPE_WIDTHS[accessor["type"]]
    stride = view.get("byteStride", width * component_bytes)
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    unpack = struct.Struct("<" + fmt * width).unpack_from
    values = []
    for i in range(accessor["count"]):
        row = unpack(binary, start + i * stride)
        values.append(row[0] if width == 1 else row)
    return values


def load_triangles(path: Path, scale: float, ground_offset: float) -> tuple[list, dict]:
    """Decode GLB triangles and return vertices in Isaac metric XYZ.

    Verified convention: raw (x,y,z) -> (s*x, s*z, ground_offset-s*y).
    """
    gltf, binary = _read_glb(path)
    triangles: list[tuple[tuple[float, float, float], ...]] = []
    raw_bounds = [[math.inf] * 3, [-math.inf] * 3]
    metric_bounds = [[math.inf] * 3, [-math.inf] * 3]
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(gltf.get("nodes", [])))))

    def visit(node_index: int, parent: Sequence[float]) -> None:
        node = gltf["nodes"][node_index]
        world = _mat_mul(parent, _node_matrix(node))
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            for primitive in mesh.get("primitives", []):
                if primitive.get("mode", 4) != 4 or "POSITION" not in primitive.get("attributes", {}):
                    continue
                raw_positions = [_transform_point(world, p) for p in _decode_accessor(gltf, binary, primitive["attributes"]["POSITION"])]
                positions = []
                for raw in raw_positions:
                    for k in range(3):
                        raw_bounds[0][k] = min(raw_bounds[0][k], raw[k])
                        raw_bounds[1][k] = max(raw_bounds[1][k], raw[k])
                    metric = (scale * raw[0], scale * raw[2], ground_offset - scale * raw[1])
                    positions.append(metric)
                    for k in range(3):
                        metric_bounds[0][k] = min(metric_bounds[0][k], metric[k])
                        metric_bounds[1][k] = max(metric_bounds[1][k], metric[k])
                indices = list(range(len(positions))) if "indices" not in primitive else _decode_accessor(gltf, binary, primitive["indices"])
                for i in range(0, len(indices) - 2, 3):
                    triangles.append((positions[indices[i]], positions[indices[i + 1]], positions[indices[i + 2]]))
        for child in node.get("children", []):
            visit(child, world)

    for root in roots:
        visit(root, _mat_identity())
    if not triangles:
        raise ValueError("GLB contains no indexed/non-indexed triangle primitives")
    return triangles, {
        "raw_bounds": raw_bounds,
        "metric_bounds": metric_bounds,
        "triangle_count": len(triangles),
        "node_count": len(gltf.get("nodes", [])),
        "mesh_count": len(gltf.get("meshes", [])),
    }


def quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    pos = (len(ordered) - 1) * q
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def _cell_center(row: int, col: int, xmin: float, ymin: float, cell: float) -> tuple[float, float]:
    return xmin + (col + 0.5) * cell, ymin + (row + 0.5) * cell


def _mark_line(mask: list[list[bool]], a: tuple[float, float], b: tuple[float, float], bounds: tuple[float, float], cell: float) -> None:
    xmin, ymin = bounds
    length = math.dist(a, b)
    steps = max(1, int(math.ceil(length / (cell * 0.35))))
    rows, cols = len(mask), len(mask[0])
    for i in range(steps + 1):
        t = i / steps
        col = int((a[0] + t * (b[0] - a[0]) - xmin) / cell)
        row = int((a[1] + t * (b[1] - a[1]) - ymin) / cell)
        if 0 <= row < rows and 0 <= col < cols:
            mask[row][col] = True


def _raster_triangle(mask: list[list[bool]], triangle: Sequence[Sequence[float]], bounds: tuple[float, float], cell: float) -> None:
    xmin, ymin = bounds
    rows, cols = len(mask), len(mask[0])
    p = [(v[0], v[1]) for v in triangle]
    min_col = max(0, int(math.floor((min(v[0] for v in p) - xmin) / cell)) - 1)
    max_col = min(cols - 1, int(math.floor((max(v[0] for v in p) - xmin) / cell)) + 1)
    min_row = max(0, int(math.floor((min(v[1] for v in p) - ymin) / cell)) - 1)
    max_row = min(rows - 1, int(math.floor((max(v[1] for v in p) - ymin) / cell)) + 1)
    if min_col > max_col or min_row > max_row:
        return
    ax, ay = p[0]
    bx, by = p[1]
    cx, cy = p[2]
    den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(den) > 1e-10:
        tolerance = 0.08
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                x, y = _cell_center(row, col, xmin, ymin, cell)
                u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den
                v = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den
                w = 1.0 - u - v
                if u >= -tolerance and v >= -tolerance and w >= -tolerance:
                    mask[row][col] = True
    # Edges also preserve thin, near-vertical walls whose XY projection is a line.
    _mark_line(mask, p[0], p[1], bounds, cell)
    _mark_line(mask, p[1], p[2], bounds, cell)
    _mark_line(mask, p[2], p[0], bounds, cell)


def dilate(mask: list[list[bool]], radius: int) -> list[list[bool]]:
    if radius <= 0:
        return [row[:] for row in mask]
    rows, cols = len(mask), len(mask[0])
    out = [[False] * cols for _ in range(rows)]
    offsets = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1) if dr * dr + dc * dc <= radius * radius]
    for row in range(rows):
        for col in range(cols):
            if mask[row][col]:
                for dr, dc in offsets:
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        out[rr][cc] = True
    return out


def largest_component(free: list[list[bool]]) -> list[tuple[int, int]]:
    rows, cols = len(free), len(free[0])
    seen: set[tuple[int, int]] = set()
    largest: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            if not free[row][col] or (row, col) in seen:
                continue
            component = []
            queue = deque([(row, col)])
            seen.add((row, col))
            while queue:
                here = queue.popleft()
                component.append(here)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = (here[0] + dr, here[1] + dc)
                    if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and free[nxt[0]][nxt[1]] and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            if len(component) > len(largest):
                largest = component
    return largest


def bfs(source: tuple[int, int], allowed: set[tuple[int, int]]) -> tuple[dict, dict]:
    distance = {source: 0}
    parent = {source: None}
    queue = deque([source])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = row + dr, col + dc
            if nxt in allowed and nxt not in distance:
                distance[nxt] = distance[(row, col)] + 1
                parent[nxt] = (row, col)
                queue.append(nxt)
    return distance, parent


def choose_start_target(component: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int], int]:
    allowed = set(component)
    center = min(component, key=lambda p: (p[0] - sum(r for r, _ in component) / len(component)) ** 2 + (p[1] - sum(c for _, c in component) / len(component)) ** 2)
    dist, _ = bfs(center, allowed)
    endpoint_a = max(dist, key=lambda p: (dist[p], -p[0], -p[1]))
    dist_a, parents = bfs(endpoint_a, allowed)
    endpoint_b = max(dist_a, key=lambda p: (dist_a[p], p[0], p[1]))
    # Cap the fixed demo problem so tabular learning is fast, but preserve a
    # meaningful route through geometry-derived free space.
    path = []
    cursor = endpoint_b
    while cursor is not None:
        path.append(cursor)
        cursor = parents[cursor]
    path.reverse()
    max_demo_steps = 42
    if len(path) - 1 > max_demo_steps:
        endpoint_b = path[max_demo_steps]
    return endpoint_a, endpoint_b, min(dist_a[endpoint_b], max_demo_steps)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_rgb_png(path: Path, pixels: list[list[tuple[int, int, int]]], scale: int = 8) -> None:
    height, width = len(pixels), len(pixels[0])
    scaled_rows = []
    for row in reversed(pixels):  # show +Y upward
        raw = bytes(channel for pixel in row for _ in range(scale) for channel in pixel)
        for _ in range(scale):
            scaled_rows.append(b"\x00" + raw)
    header = struct.pack(">IIBBBBB", width * scale, height * scale, 8, 2, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IDAT", zlib.compress(b"".join(scaled_rows), 9)) + _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def load_world_scale(world_path: Path) -> tuple[float, float]:
    response = json.loads(world_path.read_text(encoding="utf-8"))
    world = response.get("world", response)
    metadata = world["assets"]["splats"]["semantics_metadata"]
    return float(metadata["metric_scale_factor"]), float(metadata["ground_plane_offset"])


def component_boxes(mask: list[list[bool]], origin: tuple[float, float], cell: float, limit: int = 32) -> list[dict]:
    rows, cols = len(mask), len(mask[0])
    seen = set()
    boxes = []
    for row in range(rows):
        for col in range(cols):
            if not mask[row][col] or (row, col) in seen:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            points = []
            while queue:
                p = queue.popleft()
                points.append(p)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nxt = p[0] + dr, p[1] + dc
                    if 0 <= nxt[0] < rows and 0 <= nxt[1] < cols and mask[nxt[0]][nxt[1]] and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            if len(points) < 2:
                continue
            r0, r1 = min(p[0] for p in points), max(p[0] for p in points)
            c0, c1 = min(p[1] for p in points), max(p[1] for p in points)
            boxes.append({
                "center_xy_m": [round(origin[0] + (c0 + c1 + 1) * cell / 2, 4), round(origin[1] + (r0 + r1 + 1) * cell / 2, 4)],
                "size_xy_m": [round((c1 - c0 + 1) * cell, 4), round((r1 - r0 + 1) * cell, 4)],
                "source_cells": len(points),
            })
    boxes.sort(key=lambda box: box["source_cells"], reverse=True)
    return boxes[:limit]


def ensure_semantics(semantics_path: Path, collider: Path, scale: float, ground_offset: float, crop: Sequence[float], auto: bool) -> dict | None:
    """Load semantics.json, running segment_marble_collider.py first if needed."""
    stale = not semantics_path.is_file() or semantics_path.stat().st_mtime_ns < collider.stat().st_mtime_ns
    if stale and auto:
        script = Path(__file__).resolve().with_name("segment_marble_collider.py")
        command = [
            sys.executable, str(script),
            "--collider", str(collider),
            "--scale", str(scale), "--ground-offset", str(ground_offset),
            "--crop", *[str(v) for v in crop],
            "--out-dir", str(semantics_path.parent),
        ]
        print(f"SEGMENTING {' '.join(command[1:])}", file=sys.stderr)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    if not semantics_path.is_file():
        return None
    return json.loads(semantics_path.read_text(encoding="utf-8"))


def _bbox_distance(x: float, y: float, box: Sequence[float]) -> float:
    x0, x1, y0, y1 = box
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def table_approach_targets(
    tables: Sequence[dict],
    free_cells: set,
    origin: tuple[float, float],
    cell: float,
    approach_max_m: float,
) -> dict[str, dict]:
    """Pick, per detected table, the reachable free cell closest to its footprint."""
    targets: dict[str, dict] = {}
    for table in tables:
        x0, y0, _ = table["bbox_min_m"]
        x1, y1, _ = table["bbox_max_m"]
        box = (x0, x1, y0, y1)
        cx, cy = table["centroid_xyz_m"][:2]
        best = None
        for (row, col) in free_cells:
            px, py = _cell_center(row, col, origin[0], origin[1], cell)
            key = (_bbox_distance(px, py, box), math.hypot(px - cx, py - cy), row, col)
            if best is None or key < best:
                best = key
        if best is None or best[0] > approach_max_m:
            targets[table["name"]] = {
                "reachable": False,
                "table_centroid_xyz_m": table["centroid_xyz_m"],
                "table_height_m": table["height_z_m"],
                "table_area_m2": round(table["area_m2"], 3),
                "reason": "no free cell within approach_max_m of the table footprint" if best else "no free cells",
            }
            continue
        gap, centroid_distance, row, col = best
        px, py = _cell_center(row, col, origin[0], origin[1], cell)
        targets[table["name"]] = {
            "reachable": True,
            "cell": [row, col],
            "xy_m": [round(px, 5), round(py, 5)],
            "approach_gap_m": round(gap, 3),
            "centroid_distance_m": round(centroid_distance, 3),
            "table_centroid_xyz_m": table["centroid_xyz_m"],
            "table_height_m": table["height_z_m"],
            "table_footprint_xy_m": table["footprint_xy_m"],
            "table_bbox_xy_m": [round(v, 4) for v in box],
            "table_area_m2": round(table["area_m2"], 3),
        }
    return targets


def choose_demo_table(semantic_targets: dict[str, dict], allowed: set, preferred: str | None, max_demo_steps: int) -> tuple[str, tuple[int, int], int]:
    """Return (table name, start cell, steps) for the demo route.

    The start is the free cell farthest from the table's approach cell (walked
    back along the shortest path so the demo problem stays at most
    max_demo_steps long), mirroring the old free-space diameter heuristic.
    """
    def start_for(goal: tuple[int, int]) -> tuple[tuple[int, int], int]:
        dist, parents = bfs(goal, allowed)
        far = max(dist, key=lambda p: (dist[p], -p[0], -p[1]))
        path = []
        cursor = far
        while cursor is not None:
            path.append(cursor)
            cursor = parents[cursor]
        # path runs far -> goal; trim from the far end
        if len(path) - 1 > max_demo_steps:
            far = path[len(path) - 1 - max_demo_steps]
        return far, min(dist[far], max_demo_steps)

    candidates = {name: t for name, t in semantic_targets.items() if t.get("reachable")}
    if not candidates:
        raise RuntimeError("no detected table has a reachable approach cell")
    if preferred and preferred != "auto":
        if preferred not in candidates:
            raise KeyError(f"--demo-table {preferred!r} is not a reachable detected table; choices={sorted(candidates)}")
        goal = tuple(candidates[preferred]["cell"])
        start, steps = start_for(goal)
        return preferred, start, steps
    scored = []
    for name, target in candidates.items():
        goal = tuple(target["cell"])
        start, steps = start_for(goal)
        scored.append((steps, target["table_area_m2"], name, start))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    steps, _, name, start = scored[0]
    return name, start, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collider", default="artifacts/corgi-cafe/collider.glb")
    parser.add_argument("--world", default="artifacts/corgi-cafe/world.json")
    parser.add_argument("--out-dir", default="artifacts/corgi-cafe/nav")
    parser.add_argument("--cell-size", type=float, default=0.35, help="grid resolution in metres")
    parser.add_argument("--robot-radius", type=float, default=0.28, help="obstacle inflation radius in metres")
    parser.add_argument("--crop", nargs=4, type=float, metavar=("XMIN", "XMAX", "YMIN", "YMAX"), help="optional Isaac-XY crop; recorded as manual")
    parser.add_argument("--semantics", default="artifacts/corgi-cafe/semantic/semantics.json", help="output of segment_marble_collider.py")
    parser.add_argument("--no-auto-segment", action="store_true", help="do not run segment_marble_collider.py when semantics are missing/stale")
    parser.add_argument("--no-semantic-targets", action="store_true", help="ignore semantics and keep the free-space goal proxy")
    parser.add_argument("--demo-target", default="table_7", help="target id used by the task config and downstream scripts")
    parser.add_argument("--demo-table", default="auto", help="detected table instance (e.g. table_22) that --demo-target should alias; auto = longest capped demo route")
    parser.add_argument("--approach-max-m", type=float, default=1.2, help="max gap between an approach cell centre and the table footprint")
    parser.add_argument("--max-demo-steps", type=int, default=42)
    args = parser.parse_args()

    collider = Path(args.collider)
    world_path = Path(args.world)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale, ground_offset = load_world_scale(world_path)
    triangles, mesh_info = load_triangles(collider, scale, ground_offset)

    floor_z = 0.0  # metadata transform explicitly maps the reconstructed ground here
    floor_band = (-0.12, 0.18)
    obstacle_band = (0.12, 1.35)
    floor_vertices = [v for tri in triangles for v in tri if floor_band[0] <= v[2] <= floor_band[1]]
    if len(floor_vertices) < 30:
        floor_vertices = [v for tri in triangles for v in tri]
    if args.crop:
        xmin, xmax, ymin, ymax = args.crop
        crop_source = "manual_cli_crop"
    else:
        xs = [v[0] for v in floor_vertices]
        ys = [v[1] for v in floor_vertices]
        xmin, xmax = quantile(xs, 0.01), quantile(xs, 0.99)
        ymin, ymax = quantile(ys, 0.01), quantile(ys, 0.99)
        crop_source = "collider_floor_vertex_q01_q99"
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("invalid crop bounds")

    cell = args.cell_size
    cols = max(3, int(math.ceil((xmax - xmin) / cell)))
    rows = max(3, int(math.ceil((ymax - ymin) / cell)))
    # Keep the pure-Python raster bounded on unexpectedly huge reconstructions.
    if max(rows, cols) > 180:
        cell *= max(rows, cols) / 180
        cols = max(3, int(math.ceil((xmax - xmin) / cell)))
        rows = max(3, int(math.ceil((ymax - ymin) / cell)))

    floor_support = [[False] * cols for _ in range(rows)]
    obstacles = [[False] * cols for _ in range(rows)]
    floor_triangles = obstacle_triangles = 0
    for triangle in triangles:
        z_values = [v[2] for v in triangle]
        ux, uy, uz = (triangle[1][i] - triangle[0][i] for i in range(3))
        vx, vy, vz = (triangle[2][i] - triangle[0][i] for i in range(3))
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        normal_length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        horizontal = abs(nz) / normal_length >= 0.72
        mean_z = sum(z_values) / 3
        is_floor_surface = horizontal and max(z_values) >= floor_band[0] and min(z_values) <= floor_band[1] and mean_z <= 0.22
        if is_floor_surface:
            _raster_triangle(floor_support, triangle, (xmin, ymin), cell)
            floor_triangles += 1
        # A noisy fused floor triangle may straddle z=0.12; do not classify the
        # same near-horizontal support surface as both floor and obstacle.
        if not is_floor_surface and max(z_values) >= obstacle_band[0] and min(z_values) <= obstacle_band[1]:
            _raster_triangle(obstacles, triangle, (xmin, ymin), cell)
            obstacle_triangles += 1

    # One-cell floor closing tolerates cracks in the fused surface.  Obstacles
    # are then inflated by the Carter footprint.
    floor_support = dilate(floor_support, 1)
    inflation_cells = max(0, int(math.ceil(args.robot_radius / cell)))
    inflated_obstacles = dilate(obstacles, inflation_cells)
    free = [[floor_support[r][c] and not inflated_obstacles[r][c] for c in range(cols)] for r in range(rows)]
    component = largest_component(free)
    extraction_mode = "collider_floor_and_height_slice"

    if len(component) < 24:
        # Honest fallback for noisy fused meshes: use collider-derived robust
        # floor bounds, while retaining rasterized obstacle evidence.  This is
        # a geometry proxy map, not claimed to be semantic reconstruction.
        extraction_mode = "collider_bounds_plus_obstacle_proxy_fallback"
        floor_support = [[1 <= r < rows - 1 and 1 <= c < cols - 1 for c in range(cols)] for r in range(rows)]
        free = [[floor_support[r][c] and not inflated_obstacles[r][c] for c in range(cols)] for r in range(rows)]
        component = largest_component(free)
    if len(component) < 12:
        raise RuntimeError(f"no usable connected navigation island (largest={len(component)} cells); pass --crop")

    # Crop tightly to the selected walkable island, making all other islands
    # occupied/unknown.  Preserve one cell of boundary where possible.
    r0, r1 = max(0, min(r for r, _ in component) - 1), min(rows - 1, max(r for r, _ in component) + 1)
    c0, c1 = max(0, min(c for _, c in component) - 1), min(cols - 1, max(c for _, c in component) + 1)
    selected = set(component)
    occupancy = []
    raw_obstacles_cropped = []
    for r in range(r0, r1 + 1):
        occupancy.append([0 if (r, c) in selected else 1 for c in range(c0, c1 + 1)])
        raw_obstacles_cropped.append([obstacles[r][c] for c in range(c0, c1 + 1)])
    origin = (xmin + c0 * cell, ymin + r0 * cell)
    rows, cols = len(occupancy), len(occupancy[0])
    component_local = [(r - r0, c - c0) for r, c in component]

    semantics = None
    semantic_targets: dict[str, dict] = {}
    demo_table = None
    footprint_cells_closed = 0
    if not args.no_semantic_targets:
        semantics = ensure_semantics(Path(args.semantics), collider, scale, ground_offset, (xmin, xmax, ymin, ymax), not args.no_auto_segment)
    if semantics and semantics.get("tables"):
        # Close any free cell whose centre lies inside a detected table footprint.
        # Rasterised obstacles normally cover tables already; this catches thin
        # or partially reconstructed tops.
        boxes = [(t["bbox_min_m"][0], t["bbox_max_m"][0], t["bbox_min_m"][1], t["bbox_max_m"][1]) for t in semantics["tables"]]
        kept = []
        for (r, c) in component_local:
            px, py = _cell_center(r, c, origin[0], origin[1], cell)
            if any(_bbox_distance(px, py, box) == 0.0 for box in boxes):
                occupancy[r][c] = 1
                footprint_cells_closed += 1
            else:
                kept.append((r, c))
        free_local = [[occupancy[r][c] == 0 for c in range(cols)] for r in range(rows)]
        component_local = largest_component(free_local)
        for r in range(rows):
            for c in range(cols):
                if occupancy[r][c] == 0 and (r, c) not in set(component_local):
                    occupancy[r][c] = 1
        if len(component_local) < 12:
            raise RuntimeError("closing table footprints left no usable navigation island")
        allowed_local = set(component_local)
        semantic_targets = table_approach_targets(semantics["tables"], allowed_local, origin, cell, args.approach_max_m)
        demo_table, start, shortest_steps = choose_demo_table(semantic_targets, allowed_local, args.demo_table, args.max_demo_steps)
        target = tuple(semantic_targets[demo_table]["cell"])
        dist_from_start, _ = bfs(start, allowed_local)
        for name, entry in semantic_targets.items():
            if entry.get("reachable"):
                entry["shortest_path_steps"] = dist_from_start.get(tuple(entry["cell"]))
    else:
        start, target, shortest_steps = choose_start_target(component_local)

    def xy(cell_rc: tuple[int, int]) -> list[float]:
        x, y = _cell_center(cell_rc[0], cell_rc[1], origin[0], origin[1], cell)
        return [round(x, 5), round(y, 5)]

    grid_payload = {
        "schema_version": "world2work.occupancy-grid.v1",
        "rows": rows,
        "cols": cols,
        "occupied_value": 1,
        "free_value": 0,
        "cells": occupancy,
    }
    (out_dir / "occupancy_grid.json").write_text(json.dumps(grid_payload, indent=2) + "\n", encoding="utf-8")

    proxies = component_boxes(raw_obstacles_cropped, origin, cell)
    meta = {
        "schema_version": "world2work.grid-meta.v1",
        "source": {
            "kind": "World Labs Marble fused collider mesh",
            "collider": os.path.relpath(collider),
            "collider_sha256": hashlib.sha256(collider.read_bytes()).hexdigest(),
            "world": os.path.relpath(world_path),
            **mesh_info,
        },
        "transform": {
            "source_axes": "Y-up",
            "target_axes": "Isaac Z-up",
            "formula": "raw (x,y,z) -> (s*x, s*z, ground_offset - s*y)",
            "scale": scale,
            "ground_offset": ground_offset,
        },
        "extraction": {
            "mode": extraction_mode,
            "crop_source": crop_source,
            "initial_crop_xy_m": [xmin, xmax, ymin, ymax],
            "floor_z_m": floor_z,
            "floor_band_z_m": list(floor_band),
            "obstacle_band_z_m": list(obstacle_band),
            "floor_triangle_count": floor_triangles,
            "obstacle_triangle_count": obstacle_triangles,
            "robot_radius_m": args.robot_radius,
            "inflation_cells": inflation_cells,
            "semantic_table_detection": demo_table is not None,
            "semantics_source": os.path.relpath(args.semantics) if semantics else None,
            "semantic_footprint_cells_closed": footprint_cells_closed,
            "goal_note": (
                f"{args.demo_target} is the approach cell next to detected table top {demo_table}; "
                "the table itself is an occupied footprint and the robot stops beside it."
                if demo_table
                else f"{args.demo_target} is a reachable navigation goal proxy selected from the derived free-space island; the fused GLB has no object labels."
            ),
        },
        "cell_size_m": cell,
        "origin_xy_m": [origin[0], origin[1]],
        "rows": rows,
        "cols": cols,
        "indexing": "cells[row][col]; row increases +Y, col increases +X; XY is the cell centre",
        "start": {
            "cell": list(start),
            "xy_m": xy(start),
            "selection": "farthest_free_cell_from_table_approach_capped" if demo_table else "approximate_free_space_diameter_endpoint",
        },
        "targets": {
            args.demo_target: (
                {
                    "cell": list(target),
                    "xy_m": xy(target),
                    "selection": "semantic_table_approach",
                    "semantic_instance": demo_table,
                    "table_centroid_xyz_m": semantic_targets[demo_table]["table_centroid_xyz_m"],
                    "table_height_m": semantic_targets[demo_table]["table_height_m"],
                    "approach_gap_m": semantic_targets[demo_table]["approach_gap_m"],
                    "shortest_path_steps": shortest_steps,
                }
                if demo_table
                else {"cell": list(target), "xy_m": xy(target), "selection": "reachable_demo_goal_proxy", "shortest_path_steps": shortest_steps}
            )
        },
        "semantic_targets": semantic_targets,
        "stats": {
            "cells": rows * cols,
            "free_cells": sum(value == 0 for row in occupancy for value in row),
            "occupied_or_unknown_cells": sum(value == 1 for row in occupancy for value in row),
            "free_fraction": sum(value == 0 for row in occupancy for value in row) / (rows * cols),
        },
        "obstacle_proxies": proxies,
    }
    (out_dir / "grid_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    pixels = [[(246, 245, 240) if value == 0 else (35, 39, 47) for value in row] for row in occupancy]
    pixels[start[0]][start[1]] = (31, 170, 89)
    pixels[target[0]][target[1]] = (237, 125, 49)
    write_rgb_png(out_dir / "occupancy.png", pixels)
    print(json.dumps({
        "status": "GRID_OK",
        "out_dir": str(out_dir),
        "mode": extraction_mode,
        "grid": [rows, cols],
        "cell_size_m": round(cell, 4),
        "free_cells": meta["stats"]["free_cells"],
        "start": meta["start"],
        "target": meta["targets"][args.demo_target],
        "semantic_tables_reachable": sum(1 for t in semantic_targets.values() if t.get("reachable")),
        "semantic_tables_total": len(semantic_targets),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GRID_ERROR: {exc}", file=sys.stderr)
        raise
