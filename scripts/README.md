# Pipeline scripts

The repository keeps the hackathon pipeline as explicit command-line stages so each artifact can be inspected independently.

| Stage | Main scripts | Purpose |
|---|---|---|
| World generation | `generate_cafe_world.mjs`, `fetch_worldlabs_assets.mjs` | Create a Marble world and download its exports |
| Asset inspection | `inspect_worldlabs_assets.mjs`, `inspect_spz.py`, `check_worldlabs_alignment.py` | Audit mesh, splat, and coordinate alignment |
| World compilation | `segment_marble_collider.py`, `marble_to_grid.py` | Build semantic proxies and the navigation grid |
| Learning | `train_nav_qlearning.py`, `train_nav_dqn.py` | Train and export navigation policies |
| Evaluation | `eval_nav_policy.py`, `route_from_evaluation.py` | Compare policies on held-out starts and export routes |
| Isaac Sim | `isaac_nav_episode.py`, `isaac_world_robot_smoke.py` | Execute routes and record simulator telemetry |
| Demo | `make_demo_video.py`, `serve_demo.mjs` | Render the video and serve the evaluation dashboard |

Isaac Sim scripts must run with Isaac Sim's bundled Python. The remaining Python scripts can run in a regular environment created from `requirements.txt`.

RunPod helper scripts require `POD` to be set explicitly; no machine-specific endpoint is stored in the repository.
