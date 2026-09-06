#!/usr/bin/env python3
"""Evaluate untrained (random) vs trained tabular policies in the grid simulator.

Reads the artifacts written by train_nav_qlearning.py and runs N evaluation
episodes per policy from random free-cell starts with actuation noise, so the
numbers are not a replay of the training route. Emits eval_<target>.json plus
a markdown table on stdout. All numbers come from actual rollouts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import deque
from pathlib import Path


ACTIONS = [[1, 0], [0, 1], [-1, 0], [0, -1]]  # north, east, south, west (row, col)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bfs_distance(goal: tuple[int, int], free: set) -> dict:
    distance = {goal: 0}
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for dr, dc in ACTIONS:
            nxt = row + dr, col + dc
            if nxt in free and nxt not in distance:
                distance[nxt] = distance[(row, col)] + 1
                queue.append(nxt)
    return distance


def argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def rollout(start, goal, free, choose_action, rng, noise, max_steps):
    state = start
    collisions = 0
    steps = 0
    for step in range(1, max_steps + 1):
        action = choose_action(state)
        if noise > 0 and rng.random() < noise:
            action = rng.randrange(4)  # actuation slip: robot executes a random move
        dr, dc = ACTIONS[action]
        proposed = state[0] + dr, state[1] + dc
        if proposed in free:
            state = proposed
        else:
            collisions += 1
        steps = step
        if state == goal:
            return {"success": True, "steps": steps, "collisions": collisions}
    return {"success": False, "steps": steps, "collisions": collisions}


def summarize(records: list[dict]) -> dict:
    n = len(records)
    successes = [r for r in records if r["success"]]
    efficiencies = [r["shortest_steps"] / r["steps"] for r in successes if r["steps"] > 0]
    return {
        "episodes": n,
        "success_rate": round(len(successes) / n, 4),
        "collision_rate": round(sum(1 for r in records if r["collisions"] > 0) / n, 4),
        "collisions_per_episode": round(sum(r["collisions"] for r in records) / n, 4),
        "path_efficiency_mean": round(sum(efficiencies) / len(efficiencies), 4) if efficiencies else 0.0,
        "path_efficiency_over_all_episodes": round(sum(efficiencies) / n, 4),
        "mean_steps": round(sum(r["steps"] for r in records) / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-dir", default="artifacts/corgi-cafe/nav")
    parser.add_argument("--target", default="table_7")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.10, help="probability an action slips to a random move")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--isaac-starts", type=int, default=5, help="how many eval starts to export for Isaac Sim tests")
    parser.add_argument("--isaac-min-steps", type=int, default=12, help="only export Isaac starts at least this far from the goal")
    args = parser.parse_args()

    nav_dir = Path(args.nav_dir)
    grid = load_json(nav_dir / "occupancy_grid.json")
    meta = load_json(nav_dir / "grid_meta.json")
    policy_payload = load_json(nav_dir / f"policy_{args.target}.json")

    rows, cols = grid["rows"], grid["cols"]
    free = {(r, c) for r in range(rows) for c in range(cols) if grid["cells"][r][c] == grid["free_value"]}
    goal = tuple(meta["targets"][args.target]["cell"])
    demo_start = tuple(meta["start"]["cell"])
    distance = bfs_distance(goal, free)
    q_values = {tuple(int(v) for v in key.split(",")): values for key, values in policy_payload["q_values"].items()}
    max_steps = int(policy_payload["hyperparameters"]["max_steps"])

    # Evaluation starts: random reachable cells, excluding the goal and the fixed
    # demo start the learning curve was measured on. Same starts for both policies.
    candidates = sorted(s for s in distance if s != goal and s != demo_start)
    rng = random.Random(args.seed)
    starts = [rng.choice(candidates) for _ in range(args.episodes)]

    def trained(state):
        return argmax(q_values[state])

    def untrained(state):
        return rng.randrange(4)

    results = {}
    for name, chooser in (("untrained_random", untrained), ("trained_q_learning", trained)):
        episode_rng = random.Random(args.seed + 1)
        records = []
        for index, start in enumerate(starts):
            record = rollout(start, goal, free, chooser, episode_rng, args.noise, max_steps)
            record.update({"episode": index + 1, "start_cell": list(start), "shortest_steps": distance[start]})
            records.append(record)
        results[name] = {"summary": summarize(records), "episodes": records}

    origin, cell = meta["origin_xy_m"], meta["cell_size_m"]
    pose_rng = random.Random(args.seed + 2)
    isaac_starts = []
    far_starts = []
    for start in starts:
        if distance[start] >= args.isaac_min_steps and start not in far_starts:
            far_starts.append(start)
    for start in far_starts[: args.isaac_starts]:
        # Continuous held-out poses for Isaac Sim: sub-cell offset plus random heading,
        # which the grid policy never saw during training.
        x = origin[0] + (start[1] + 0.5) * cell + pose_rng.uniform(-0.3, 0.3) * cell
        y = origin[1] + (start[0] + 0.5) * cell + pose_rng.uniform(-0.3, 0.3) * cell
        isaac_starts.append({
            "cell": list(start),
            "xy_m": [round(x, 4), round(y, 4)],
            "yaw_deg": round(pose_rng.uniform(-30.0, 30.0), 1),
            "shortest_steps": distance[start],
            "shortest_path_m": round(distance[start] * cell, 3),
        })

    payload = {
        "schema_version": "world2work.nav-eval.v1",
        "target_id": args.target,
        "simulator": "grid_sim_from_marble_collider",
        "protocol": {
            "episodes_per_policy": args.episodes,
            "start_selection": "random reachable free cells, excluding goal and the fixed demo start; identical starts for both policies",
            "actuation_noise": args.noise,
            "max_steps": max_steps,
            "seed": args.seed,
            "collision_definition": "attempted move into an occupied cell or outside the grid",
            "path_efficiency_definition": "BFS shortest steps / executed steps, successful episodes only",
            "caveat": "training used curriculum starts across the same free-space island, so grid starts are not strictly unseen; Isaac Sim tests use continuous held-out poses (isaac_test_starts)",
        },
        "policies": {name: value["summary"] for name, value in results.items()},
        "isaac_test_starts": isaac_starts,
        "records": {name: value["episodes"] for name, value in results.items()},
    }
    output = nav_dir / f"eval_{args.target}.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    u, t = payload["policies"]["untrained_random"], payload["policies"]["trained_q_learning"]
    print(f"grid-sim evaluation, target={args.target}, N={args.episodes} per policy, noise={args.noise}")
    print("| Metric | Untrained (random) | Trained (Q-learning) |")
    print("|---|---|---|")
    print(f"| Success rate | {u['success_rate']:.0%} | {t['success_rate']:.0%} |")
    print(f"| Collision rate (episodes with >=1 collision) | {u['collision_rate']:.0%} | {t['collision_rate']:.0%} |")
    print(f"| Collisions / episode | {u['collisions_per_episode']} | {t['collisions_per_episode']} |")
    print(f"| Path efficiency (successes) | {u['path_efficiency_mean']:.2f} | {t['path_efficiency_mean']:.2f} |")
    print(f"| Mean steps | {u['mean_steps']} | {t['mean_steps']} |")
    print(f"written: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"EVAL_ERROR: {exc}", file=sys.stderr)
        raise
