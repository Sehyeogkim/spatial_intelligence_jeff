# World2Work browser demo

Dependency-free static dashboard for the World2Work navigation demo:

1. A user café photo is reconstructed as a World Labs Marble world.
2. The Marble fused collider is converted to an occupancy grid.
3. A **tabular Q-learning high-level policy** learns a route to `table_7`.
4. The page renders the learned route as a **kinematic browser replay** of Nova Carter carrying a tray-fixed coffee payload.

The browser replay is not presented as rigid-body physics. A separate **Actual Isaac Sim rollout (Nova Carter)** card loads `assets/world2work_table7.mp4` when that rendered video is present. If it is absent or invalid, the card shows a non-broken pending state with a retry action.

## Run

Serve the repository root so the page can read local artifacts:

```bash
cd /Users/jeff/project/spatial_intelligence_jeff
python3 -m http.server 4173
```

Open <http://localhost:4173/web-demo/>.

## Local artifact contract

The page loads these files from `artifacts/corgi-cafe/nav/`:

- `occupancy_grid.json`
- `grid_meta.json`
- `learning_curve.json`
- `policy_table_7.json`
- `route_table_7.json`
- `evaluation_summary.json`

The simulator video lives outside that data contract at `assets/world2work_table7.mp4`.

Missing navigation or training artifacts use an explicitly labelled visual preview. Missing evaluation data never invents metric values: Success Rate, Collision Rate, Path Efficiency, and Untrained/Trained results remain `—` until `evaluation_summary.json` exists.

## Console hooks

```js
await window.world2WorkDemo.run();
window.world2WorkDemo.reset();
window.world2WorkDemo.getEpisode();
window.world2WorkDemo.getArtifacts();
```
