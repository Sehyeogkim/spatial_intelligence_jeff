# World2Work evaluation dashboard

This dependency-free dashboard packages the evidence needed to inspect the hackathon result:

- the Marble-derived occupancy grid and learned route;
- the 4,000-episode learning curve and value heatmap;
- the held-out random-versus-trained evaluation;
- a kinematic route replay; and
- the recorded Isaac Sim Nova Carter rollout.

## Run

From the repository root:

```bash
make demo
```

Then open <http://127.0.0.1:4173/web-demo/>.

The dashboard is self-contained. Its curated JSON inputs live in `web-demo/data/`, and its public media lives in `web-demo/assets/`; it does not require private World Labs exports or a running Isaac Sim instance.

## Data contract

| File | Purpose |
|---|---|
| `occupancy_grid.json` | Marble-derived navigation grid |
| `grid_meta.json` | Coordinate transform, targets, and map metadata |
| `learning_curve.json` | Aggregate training history |
| `policy_table_7.json` | Learned tabular policy |
| `route_table_7.json` | Exported Carter waypoints |
| `evaluation_summary.json` | Held-out policy comparison |

Missing data never produces invented metrics. The UI labels fallback previews explicitly and leaves unavailable measurements blank.

## Console hooks

```js
await window.world2WorkDemo.run();
window.world2WorkDemo.reset();
window.world2WorkDemo.getEpisode();
window.world2WorkDemo.getArtifacts();
```
