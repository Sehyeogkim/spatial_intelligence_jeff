#!/usr/bin/env python3
"""Turn held-out evaluation rollouts into route files isaac_nav_episode.py can drive.

Picks the first held-out start configuration where the untrained (random) and
trained policies were evaluated from the same cell, and writes both rollouts in
the world2work.nav-route.v1 schema so Isaac Sim can render the matched pair
with the same camera and start pose:

  artifacts/corgi-cafe/nav/route_holdout_trained.json
  artifacts/corgi-cafe/nav/route_holdout_untrained.json

Collision steps in the random rollout (the robot stays in the same cell) are
kept as a short bump toward the blocked neighbour so the wall contact is
visible on camera; the bump target is a point 30 % into the blocked cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cell_xy(cell, meta) -> list[float]:
    origin, size = meta["origin_xy_m"], meta["cell_size_m"]
    return [round(origin[0] + (cell[1] + 0.5) * size, 5), round(origin[1] + (cell[0] + 0.5) * size, 5)]


def build_route(episode: dict, meta: dict, free: set, target_id: str, policy_label: str, bump: float) -> dict:
    cells = [tuple(c) for c in episode["visited_cells"]]
    size = meta["cell_size_m"]
    waypoints = [cell_xy(cells[0], meta)]
    route_cells = [list(cells[0])]
    collisions = 0
    for previous, current in zip(cells, cells[1:]):
        if current == previous:
            # Collision: the policy tried to leave the cell and was blocked. We do not know
            # which neighbour it hit, so bump toward the nearest occupied neighbour.
            blocked = [(dr, dc) for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)) if (previous[0] + dr, previous[1] + dc) not in free]
            if blocked:
                dr, dc = blocked[collisions % len(blocked)]
                x, y = cell_xy(previous, meta)
                waypoints.append([round(x + dc * size * bump, 5), round(y + dr * size * bump, 5)])
                waypoints.append([x, y])
                route_cells.extend([list(previous), list(previous)])
            collisions += 1
            continue
        waypoints.append(cell_xy(current, meta))
        route_cells.append(list(current))
    goal = tuple(meta["targets"][target_id]["cell"])
    return {
        "schema_version": "world2work.nav-route.v1",
        "target_id": target_id,
        "policy_file": None,
        "policy_label": policy_label,
        "source": "evaluation_summary.json held-out rollout",
        "held_out_episode": episode["episode"],
        "initial_orientation": episode.get("initial_orientation"),
        "success": bool(episode["success"]),
        "termination": episode.get("termination"),
        "cells": route_cells,
        "waypoints_xy_m": waypoints,
        "waypoints_xyz_m": [[p[0], p[1], 0.0] for p in waypoints],
        "control_waypoints_xy_m": waypoints,
        "start": {"cell": list(cells[0]), "xy_m": cell_xy(cells[0], meta)},
        "target": {"cell": list(goal), "xy_m": meta["targets"][target_id]["xy_m"]},
        "steps": len(cells) - 1,
        "collision_actions": episode.get("collision_actions", collisions),
        "shortest_path_steps": episode["shortest_path_steps"],
        "path_length_m": round(sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(waypoints, waypoints[1:])), 5),
        "is_shortest": len(cells) - 1 == episode["shortest_path_steps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-dir", default="artifacts/corgi-cafe/nav")
    parser.add_argument("--target", default="table_7")
    parser.add_argument("--pair-index", type=int, default=0, help="which matched held-out start to export")
    parser.add_argument("--bump", type=float, default=0.3)
    args = parser.parse_args()

    nav = Path(args.nav_dir)
    meta = load_json(nav / "grid_meta.json")
    grid = load_json(nav / "occupancy_grid.json")
    evaluation = load_json(nav / "evaluation_summary.json")
    free = {(r, c) for r in range(grid["rows"]) for c in range(grid["cols"]) if grid["cells"][r][c] == grid["free_value"]}
    policies = evaluation["policies"]
    untrained_key = next(k for k in policies if "untrained" in k or "random" in k)
    trained_key = next(k for k in policies if k != untrained_key)
    trained_by_start = {tuple(e["start_cell"]): e for e in policies[trained_key]["episodes"]}
    pairs = [(e, trained_by_start[tuple(e["start_cell"])]) for e in policies[untrained_key]["episodes"] if tuple(e["start_cell"]) in trained_by_start]
    if not pairs:
        raise RuntimeError("no held-out start shared by both policies")
    untrained_ep, trained_ep = pairs[min(args.pair_index, len(pairs) - 1)]

    outputs = {}
    for label, episode in (("untrained", untrained_ep), ("trained", trained_ep)):
        route = build_route(episode, meta, free, args.target, label, args.bump)
        path = nav / f"route_holdout_{label}.json"
        path.write_text(json.dumps(route, indent=2) + "\n", encoding="utf-8")
        outputs[label] = {"file": str(path), "waypoints": len(route["waypoints_xy_m"]), "success": route["success"],
                          "collisions": route["collision_actions"], "path_length_m": route["path_length_m"]}
    print(json.dumps({"status": "ROUTES_OK", "start_cell": untrained_ep["start_cell"], "initial_orientation": untrained_ep.get("initial_orientation"),
                      "shortest_path_steps": untrained_ep["shortest_path_steps"], "routes": outputs}, indent=2))


if __name__ == "__main__":
    main()
