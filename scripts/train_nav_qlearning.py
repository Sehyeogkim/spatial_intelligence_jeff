#!/usr/bin/env python3
"""Train and export a deterministic 4-action tabular navigation policy."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import random
import struct
import sys
import zlib
from collections import deque
from pathlib import Path


ACTIONS = [
    {"id": 0, "name": "north", "delta_rc": [1, 0]},
    {"id": 1, "name": "east", "delta_rc": [0, 1]},
    {"id": 2, "name": "south", "delta_rc": [-1, 0]},
    {"id": 3, "name": "west", "delta_rc": [0, -1]},
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bfs_distance(goal: tuple[int, int], free: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for action in ACTIONS:
            dr, dc = action["delta_rc"]
            nxt = row + dr, col + dc
            if nxt in free and nxt not in distance:
                distance[nxt] = distance[(row, col)] + 1
                queue.append(nxt)
    return distance


def transition(state: tuple[int, int], action_id: int, free: set[tuple[int, int]]) -> tuple[tuple[int, int], bool]:
    dr, dc = ACTIONS[action_id]["delta_rc"]
    proposed = state[0] + dr, state[1] + dc
    return (proposed, False) if proposed in free else (state, True)


def argmax(values: list[float]) -> int:
    # Stable action ordering makes ties deterministic.
    return max(range(len(values)), key=lambda index: (values[index], -index))


def greedy_route(start: tuple[int, int], goal: tuple[int, int], q: dict, free: set, limit: int) -> tuple[list, bool, str]:
    route = [start]
    state = start
    visits = {start: 1}
    for _ in range(limit):
        if state == goal:
            return route, True, "goal"
        ranked = sorted(range(4), key=lambda a: (q[state][a], -a), reverse=True)
        moved = False
        for action in ranked:
            nxt, collision = transition(state, action, free)
            if not collision and visits.get(nxt, 0) < 2:
                state = nxt
                moved = True
                break
        if not moved:
            return route, False, "policy_loop_or_dead_end"
        route.append(state)
        visits[state] = visits.get(state, 0) + 1
    return route, state == goal, "limit"


def simplify_collinear(points: list[list[float]]) -> list[list[float]]:
    if len(points) <= 2:
        return points
    result = [points[0]]
    for index in range(1, len(points) - 1):
        before, here, after = points[index - 1], points[index], points[index + 1]
        first = (round(here[0] - before[0], 8), round(here[1] - before[1], 8))
        second = (round(after[0] - here[0], 8), round(after[1] - here[1], 8))
        if first != second:
            result.append(here)
    result.append(points[-1])
    return result


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_rgb_png(path: Path, pixels: list[list[tuple[int, int, int]]], scale: int = 8) -> None:
    height, width = len(pixels), len(pixels[0])
    rows = []
    for row in reversed(pixels):
        raw = bytes(channel for pixel in row for _ in range(scale) for channel in pixel)
        rows.extend([b"\x00" + raw] * scale)
    ihdr = struct.pack(">IIBBBBB", width * scale, height * scale, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + png_chunk(b"IEND", b""))


def colour_map(t: float) -> tuple[int, int, int]:
    # Blue -> cyan -> yellow without third-party plotting packages.
    t = min(1.0, max(0.0, t))
    hue = (0.64 * (1.0 - t))
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.96)
    return int(255 * r), int(255 * g), int(255 * b)


def select_holdout_configurations(
    reachable: list[tuple[int, int]],
    distance: dict[tuple[int, int], int],
    fixed_start: tuple[int, int],
    goal: tuple[int, int],
    count: int,
    seed: int,
) -> list[dict]:
    """Select deterministic, spatially varied evaluation start configurations.

    These cells are selected before learning and excluded from curriculum episode
    starts.  A training trajectory can still pass through one, so the claim is
    specifically "held-out episode start", not an unseen-state claim.
    """
    eligible = [state for state in reachable if state not in (fixed_start, goal) and distance[state] >= 3]
    if len(eligible) < count:
        eligible = [state for state in reachable if state not in (fixed_start, goal)]
    if len(eligible) < count:
        raise RuntimeError(f"need {count} held-out starts, but only {len(eligible)} are available")

    # Round-robin across BFS-distance buckets avoids evaluating only easy cells
    # near the goal or a single patch of the map.
    buckets: dict[int, list[tuple[int, int]]] = {}
    for state in eligible:
        buckets.setdefault(distance[state], []).append(state)
    rng = random.Random(seed)
    for states in buckets.values():
        states.sort()
        rng.shuffle(states)
    ordered = []
    distances = sorted(buckets, reverse=True)
    while len(ordered) < count:
        changed = False
        for value in distances:
            if buckets[value]:
                ordered.append(buckets[value].pop())
                changed = True
                if len(ordered) == count:
                    break
        if not changed:
            break
    orientation_offset = rng.randrange(4)
    return [
        {
            "episode": index + 1,
            "start_cell": list(state),
            "initial_orientation": ACTIONS[(index + orientation_offset) % 4]["name"],
            "shortest_path_steps": distance[state],
        }
        for index, state in enumerate(ordered)
    ]


def evaluate_policy(
    policy_name: str,
    configurations: list[dict],
    q: dict[tuple[int, int], list[float]],
    goal: tuple[int, int],
    free: set[tuple[int, int]],
    seed: int,
) -> dict:
    """Run an actual discrete rollout for every held-out start configuration."""
    rng = random.Random(seed)
    action_for_name = {action["name"]: action["id"] for action in ACTIONS}
    results = []
    total_actions = 0
    total_collisions = 0
    collision_episodes = 0
    efficiencies = []
    successes = 0
    for config in configurations:
        state = tuple(config["start_cell"])
        heading = action_for_name[config["initial_orientation"]]
        initial_heading = heading
        shortest = int(config["shortest_path_steps"])
        max_steps = max(60, min(300, shortest * 6))
        collisions = 0
        quarter_turns = 0
        visited_cells = [list(state)]
        success = False
        for step in range(1, max_steps + 1):
            if policy_name == "untrained_random":
                action = rng.randrange(4)
            elif policy_name == "trained_greedy":
                action = argmax(q[state])
            else:
                raise ValueError(f"unknown policy {policy_name}")
            turn = abs(action - heading)
            quarter_turns += min(turn, 4 - turn)
            heading = action
            nxt, collision = transition(state, action, free)
            collisions += int(collision)
            state = nxt
            visited_cells.append(list(state))
            if state == goal:
                success = True
                break
        actual_steps = step
        total_actions += actual_steps
        total_collisions += collisions
        collision_episodes += int(collisions > 0)
        successes += int(success)
        efficiency = shortest / max(actual_steps, shortest) if success else None
        if efficiency is not None:
            efficiencies.append(efficiency)
        results.append({
            "episode": config["episode"],
            "start_cell": config["start_cell"],
            "initial_orientation": ACTIONS[initial_heading]["name"],
            "final_orientation": ACTIONS[heading]["name"],
            "shortest_path_steps": shortest,
            "actual_steps": actual_steps,
            "success": success,
            "collision_actions": collisions,
            "quarter_turns": quarter_turns,
            "path_efficiency": round(efficiency, 6) if efficiency is not None else None,
            "termination": "goal" if success else "timeout",
            "visited_cells": visited_cells,
        })
    count = len(results)
    return {
        "policy": "uniform random cardinal action, no learned parameters" if policy_name == "untrained_random" else "strict argmax over the learned Q table",
        "episodes": results,
        "metrics": {
            "success_rate": round(successes / count, 6),
            "collision_rate": round(total_collisions / max(1, total_actions), 6),
            "path_efficiency": round(sum(efficiencies) / len(efficiencies), 6) if efficiencies else None,
            "successful_episodes": successes,
            "total_episodes": count,
            "total_actions": total_actions,
            "collision_actions": total_collisions,
            "collision_episode_rate": round(collision_episodes / count, 6),
            "mean_quarter_turns": round(sum(item["quarter_turns"] for item in results) / count, 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-dir", default="artifacts/corgi-cafe/nav")
    parser.add_argument("--target", default="table_7")
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--alpha", type=float, default=0.34)
    parser.add_argument("--gamma", type=float, default=0.97)
    args = parser.parse_args()

    nav_dir = Path(args.nav_dir)
    grid = load_json(nav_dir / "occupancy_grid.json")
    meta = load_json(nav_dir / "grid_meta.json")
    cells = grid["cells"]
    rows, cols = grid["rows"], grid["cols"]
    free = {(row, col) for row in range(rows) for col in range(cols) if cells[row][col] == grid["free_value"]}
    start = tuple(meta["start"]["cell"])
    if args.target not in meta["targets"]:
        raise KeyError(f"unknown target {args.target!r}; choices={sorted(meta['targets'])}")
    goal = tuple(meta["targets"][args.target]["cell"])
    distance = bfs_distance(goal, free)
    if start not in distance:
        raise RuntimeError("start is not connected to target")

    rng = random.Random(args.seed)
    q = {state: [0.0, 0.0, 0.0, 0.0] for state in free}
    reachable = sorted(distance)
    shortest_steps = distance[start]
    holdout_count = 20
    holdout_seed = args.seed + 1701
    holdout_configs = select_holdout_configurations(reachable, distance, start, goal, holdout_count, holdout_seed)
    holdout_cells = {tuple(config["start_cell"]) for config in holdout_configs}
    max_steps = max(80, min(500, shortest_steps * 8))
    episodes = []
    rolling = []
    success_window = deque(maxlen=100)
    reward_window = deque(maxlen=100)

    experience_path = nav_dir / "training_experience.jsonl"
    experience_tmp = nav_dir / ".training_experience.jsonl.tmp"
    transition_count = 0
    with experience_tmp.open("w", encoding="utf-8") as experience_file:
        for episode in range(args.episodes):
            progress = episode / max(1, args.episodes - 1)
            epsilon = max(0.025, 0.92 * math.exp(-5.6 * progress))
            # Reverse-curriculum episodes train useful values throughout the same
            # connected Marble-derived island; alternating fixed starts makes the
            # reported success curve directly relevant to the demo route. Held-out
            # cells are never selected as episode starts.
            if episode % 2 == 0:
                state = start
            else:
                curriculum_radius = max(4, int(shortest_steps * min(1.0, 0.12 + 1.2 * progress)))
                candidates = [s for s in reachable if 1 <= distance[s] <= curriculum_radius and s not in holdout_cells]
                fallback_candidates = [s for s in reachable if s not in holdout_cells and s != goal]
                state = rng.choice(candidates or fallback_candidates)
            episode_start = state
            total_reward = 0.0
            visited = {state}
            success = False
            steps = 0
            for step in range(1, max_steps + 1):
                previous = state
                if rng.random() < epsilon:
                    action = rng.randrange(4)
                else:
                    action = argmax(q[state])
                nxt, collision = transition(state, action, free)
                if collision:
                    reward = -1.0
                else:
                    potential_gain = distance[state] - distance[nxt]
                    reward = -0.035 + 0.28 * potential_gain
                    if nxt in visited:
                        reward -= 0.04
                    if nxt == goal:
                        reward += 12.0
                success = nxt == goal
                done = success or step == max_steps
                best_next = max(q[nxt])
                target_value = reward if done else reward + args.gamma * best_next
                q[state][action] += args.alpha * (target_value - q[state][action])
                experience = {
                    "episode": episode + 1,
                    "t": step - 1,
                    "state": list(previous),
                    "action": action,
                    "action_name": ACTIONS[action]["name"],
                    "reward": round(reward, 6),
                    "next_state": list(nxt),
                    "done": done,
                    "success": success,
                    "collision": collision,
                    "start_kind": "fixed" if episode_start == start else "curriculum",
                }
                experience_file.write(json.dumps(experience, separators=(",", ":")) + "\n")
                transition_count += 1
                total_reward += reward
                state = nxt
                visited.add(state)
                steps = step
                if success:
                    break
            # Curves use fixed-start episodes only, avoiding curriculum inflation.
            if episode_start == start:
                success_window.append(1 if success else 0)
                reward_window.append(total_reward)
            episodes.append({
                "episode": episode + 1,
                "reward": round(total_reward, 6),
                "success": success,
                "steps": steps,
                "start_kind": "fixed" if episode_start == start else "curriculum",
                "epsilon": round(epsilon, 6),
            })
            if (episode + 1) % 50 == 0:
                rolling.append({
                    "episode": episode + 1,
                    "mean_reward_fixed_start": round(sum(reward_window) / max(1, len(reward_window)), 6),
                    "success_rate_fixed_start": round(sum(success_window) / max(1, len(success_window)), 6),
                })
    experience_tmp.replace(experience_path)

    route, route_success, route_reason = greedy_route(start, goal, q, free, max(rows * cols, shortest_steps * 4))
    if not route_success:
        raise RuntimeError(f"trained greedy policy failed validation: {route_reason}; route_len={len(route)}")

    policy = [[-1] * cols for _ in range(rows)]
    q_values = {}
    for state in sorted(free):
        policy[state[0]][state[1]] = argmax(q[state])
        q_values[f"{state[0]},{state[1]}"] = [round(v, 6) for v in q[state]]
    policy_payload = {
        "schema_version": "world2work.tabular-q-policy.v1",
        "algorithm": "tabular_q_learning",
        "target_id": args.target,
        "seed": args.seed,
        "actions": ACTIONS,
        "rows": rows,
        "cols": cols,
        "policy": policy,
        "q_values": q_values,
        "hyperparameters": {
            "episodes": args.episodes,
            "alpha": args.alpha,
            "gamma": args.gamma,
            "epsilon_initial": 0.92,
            "epsilon_final": 0.025,
            "max_steps": max_steps,
            "reward": {"step": -0.035, "progress_per_bfs_cell": 0.28, "collision": -1.0, "goal_bonus": 12.0, "revisit": -0.04},
        },
        "validation": {"greedy_success": True, "greedy_steps": len(route) - 1, "shortest_path_steps": shortest_steps},
    }
    (nav_dir / f"policy_{args.target}.json").write_text(json.dumps(policy_payload, indent=2) + "\n", encoding="utf-8")

    learning_payload = {
        "schema_version": "world2work.learning-curve.v1",
        "algorithm": "tabular_q_learning",
        "target_id": args.target,
        "seed": args.seed,
        "episodes": episodes,
        "rolling": rolling,
        "summary": {
            "fixed_start_success_rate_last_100": rolling[-1]["success_rate_fixed_start"] if rolling else None,
            "greedy_success": route_success,
            "greedy_steps": len(route) - 1,
            "shortest_path_steps": shortest_steps,
        },
    }
    (nav_dir / "learning_curve.json").write_text(json.dumps(learning_payload, indent=2) + "\n", encoding="utf-8")

    origin = meta["origin_xy_m"]
    cell_size = meta["cell_size_m"]
    waypoints = [[round(origin[0] + (col + 0.5) * cell_size, 5), round(origin[1] + (row + 0.5) * cell_size, 5)] for row, col in route]
    route_payload = {
        "schema_version": "world2work.nav-route.v1",
        "target_id": args.target,
        "policy_file": f"policy_{args.target}.json",
        "success": route_success,
        "termination": route_reason,
        "cells": [list(p) for p in route],
        "waypoints_xy_m": waypoints,
        "waypoints_xyz_m": [[p[0], p[1], 0.0] for p in waypoints],
        "control_waypoints_xy_m": simplify_collinear(waypoints),
        "start": {"cell": list(start), "xy_m": meta["start"]["xy_m"]},
        "target": {"cell": list(goal), "xy_m": meta["targets"][args.target]["xy_m"]},
        "steps": len(route) - 1,
        "shortest_path_steps": shortest_steps,
        "path_length_m": round((len(route) - 1) * cell_size, 5),
        "is_shortest": len(route) - 1 == shortest_steps,
    }
    (nav_dir / f"route_{args.target}.json").write_text(json.dumps(route_payload, indent=2) + "\n", encoding="utf-8")

    untrained_eval_seed = args.seed + 2909
    untrained_evaluation = evaluate_policy("untrained_random", holdout_configs, q, goal, free, untrained_eval_seed)
    trained_evaluation = evaluate_policy("trained_greedy", holdout_configs, q, goal, free, untrained_eval_seed)
    untrained_metrics = untrained_evaluation["metrics"]
    trained_metrics = trained_evaluation["metrics"]
    evaluation_payload = {
        "schema_version": "world2work.policy-evaluation.v1",
        "target_id": args.target,
        "seed": args.seed,
        "evaluation_seed": untrained_eval_seed,
        "holdout": {
            "episodes": len(holdout_configs),
            "selection_seed": holdout_seed,
            "cells_excluded_from_training_episode_starts": True,
            "unseen_state_claim": False,
            "note": "Training rollouts may traverse these cells. The held-out unit is the episode start cell/orientation configuration.",
            "configurations": holdout_configs,
        },
        "metric_definitions": {
            "success_rate": "successful episodes / all held-out episodes",
            "collision_rate": "actions attempting to enter occupied or out-of-bounds cells / all actions",
            "path_efficiency": "mean(shortest_path_steps / max(actual_steps, shortest_path_steps)) over successful episodes only; null when there are no successes",
            "orientation": "initial cardinal heading for the downstream controller; the high-level grid policy selects absolute cardinal moves and is orientation-invariant",
        },
        "policies": {
            "untrained_random": untrained_evaluation,
            "trained_greedy": trained_evaluation,
        },
        "comparison": {
            "success_rate_delta": round(trained_metrics["success_rate"] - untrained_metrics["success_rate"], 6),
            "collision_rate_delta": round(trained_metrics["collision_rate"] - untrained_metrics["collision_rate"], 6),
            "path_efficiency_delta": (
                round(trained_metrics["path_efficiency"] - untrained_metrics["path_efficiency"], 6)
                if trained_metrics["path_efficiency"] is not None and untrained_metrics["path_efficiency"] is not None
                else None
            ),
        },
    }
    (nav_dir / "evaluation_summary.json").write_text(json.dumps(evaluation_payload, indent=2) + "\n", encoding="utf-8")

    values = [max(q[state]) for state in free]
    low, high = min(values), max(values)
    pixels = []
    route_set = set(route)
    for row in range(rows):
        pixel_row = []
        for col in range(cols):
            state = row, col
            if state not in free:
                colour = (35, 39, 47)
            else:
                normalized = (max(q[state]) - low) / max(1e-9, high - low)
                colour = colour_map(normalized)
                if state in route_set:
                    colour = tuple(min(255, int(channel * 0.45 + accent * 0.55)) for channel, accent in zip(colour, (255, 255, 255)))
            if state == start:
                colour = (31, 170, 89)
            elif state == goal:
                colour = (237, 125, 49)
            pixel_row.append(colour)
        pixels.append(pixel_row)
    write_rgb_png(nav_dir / "value_heatmap.png", pixels)

    print(json.dumps({
        "status": "QLEARNING_OK",
        "target": args.target,
        "episodes": args.episodes,
        "last_100_fixed_start_success_rate": learning_payload["summary"]["fixed_start_success_rate_last_100"],
        "greedy_steps": len(route) - 1,
        "shortest_path_steps": shortest_steps,
        "is_shortest": route_payload["is_shortest"],
        "training_transitions": transition_count,
        "held_out_episodes": len(holdout_configs),
        "untrained_metrics": untrained_metrics,
        "trained_metrics": trained_metrics,
        "route": str(nav_dir / f"route_{args.target}.json"),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QLEARNING_ERROR: {exc}", file=sys.stderr)
        raise
