#!/usr/bin/env python3
"""Minimal headless physics smoke test for Isaac Sim 6.0.1."""

from __future__ import annotations

import json
import traceback
from pathlib import Path


OUTPUT_PATH = Path("/isaac-sim/output/isaac_smoke_result.json")
STEP_COUNT = 120


def _write_result(payload: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    # SimulationApp must be created before importing other Isaac Sim modules.
    from isaacsim import SimulationApp

    simulation_app = None
    try:
        simulation_app = SimulationApp({"headless": True})

        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/SmokeCube",
                name="smoke_cube",
                position=np.array([0.0, 0.0, 1.0]),
                size=0.25,
                color=np.array([0.2, 0.6, 1.0]),
                mass=1.0,
            )
        )

        world.reset()
        for _ in range(STEP_COUNT):
            world.step(render=False)

        position, orientation = cube.get_world_pose()
        result = {
            "ok": True,
            "isaac_sim_version": "6.0.1",
            "steps": STEP_COUNT,
            "cube": {
                "prim_path": "/World/SmokeCube",
                "position_m": [float(value) for value in position],
                "orientation_wxyz": [float(value) for value in orientation],
            },
        }
        _write_result(result)
        print(f"ISAAC_SMOKE_OK {json.dumps(result, sort_keys=True)}", flush=True)
    except Exception as exc:
        failure = {
            "ok": False,
            "isaac_sim_version": "6.0.1",
            "steps_requested": STEP_COUNT,
            "error": f"{type(exc).__name__}: {exc}",
        }
        try:
            _write_result(failure)
        except Exception:
            traceback.print_exc()
        print(f"ISAAC_SMOKE_FAILED {json.dumps(failure, sort_keys=True)}", flush=True)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
