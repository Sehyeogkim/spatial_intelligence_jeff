# World Labs API Capability Audit

Audit date: 2026-09-05  
Generated world: [Robot Capability Audit Room](https://marble.worldlabs.ai/world/e91e1761-39ec-40cd-964b-84c0d9d637cc)  
Model: `marble-1.1`

## Executive decision

World Labs is sufficient as a generated **static visual world plus collision/raycast surface**, but its output is not sufficient by itself for object-level robot manipulation data.

- Green: rendering, spatial scale, ground alignment, raycast, surface queries, robot placement, navigation, and synthetic camera capture using cameras that we create.
- Yellow: labeling a few static generated objects by adding our own 2D-to-3D semantic segmentation layer.
- Red: treating a generated apple, table, chair, or crate as an independently movable object. They are fused into one collider mesh and have no object IDs or semantic metadata.

The hackathon architecture should therefore use a World Labs world as the static scene, add separately instantiated task objects with known IDs/transforms, and generate robot/camera/action state in our own runtime.

## Test generation

The prompt explicitly requested a clean room containing one table, one red chair, one blue crate, and one apple, separated from one another to make object boundaries easy to detect.

- API request accepted: `200 OK`
- Actual generation time: approximately 16 minutes 21 seconds
- Cost: 1,580 credits
  - Text-to-pano: 80
  - World generation: 1,500
- Visual inspection: the requested table, chair, crate, and apple are present and visually distinct.

The observed generation time was much longer than the documented approximately five-minute estimate. World generation should not be placed inside a live robot episode loop; worlds should be generated ahead of the interactive demo.

## Actual outputs

| Asset | Actual size | Actual contents |
|---|---:|---|
| 100k SPZ | 1,160,936 bytes / 1.107 MiB | 98,304 Gaussians |
| 500k SPZ | 7,373,728 bytes / 7.032 MiB | 500,000 Gaussians |
| Full-resolution SPZ | 28,121,210 bytes / 26.818 MiB | 1,920,000 Gaussians |
| Collider GLB | 6,813,884 bytes / 6.498 MiB | 131,004 vertices / 262,040 triangles |
| Panorama PNG | 9,407,271 bytes / 8.971 MiB | 4608 x 2304, 2:1 equirectangular |
| 100k PLY export | 5,505,385 bytes / 5.250 MiB | 98,304 vertices, 14 float properties |

All returned SPZ files were legacy SPZ version 2, gzip-compressed, SH degree 0, without vendor extensions.

An HQ mesh export was accepted successfully and remained `IN_PROGRESS` after 20 minutes of polling. The API documents that one 3,500-credit export run produces both the textured `hq_mesh_url` and vertex-colored `full_res_mesh_url`, but those two files were not yet available for direct inspection at the audit cutoff. The export operation and downloader are saved so the check can resume without submitting or charging for a second export.

## SPZ point data

The official Niantic SPZ decoder successfully exposed the following arrays for every Gaussian:

- `positions`: `N x 3`
- log-space `scales`: `N x 3`
- quaternion `rotations`: `N x 4`
- alpha logits: `N`
- base RGB colors: `N x 3`
- spherical harmonics: supported by the format, but this world used degree 0 and therefore had zero additional SH coefficients

No decoded array exists for semantic class, object ID, instance ID, movability, affordance, or joints.

The synchronous 100k PLY export contained exactly these properties:

```text
x y z
f_dc_0 f_dc_1 f_dc_2
opacity
scale_0 scale_1 scale_2
rot_0 rot_1 rot_2 rot_3
```

No hidden semantic property appeared in the PLY header.

## Collider GLB structure

- Scene count: 1
- Nodes: 2, named only `world` and `geometry_0`
- Meshes/primitives: 1 / 1
- Vertices: 131,004
- Triangles: 262,040
- Materials/textures/images: 0 / 0 / 0
- Cameras/skins/animations: 0 / 0 / 0
- Only extras metadata: `{ "processed": true }`
- Semantic-like keys: 0

The mesh had nine connected components, but the main component contained 261,896 of 262,040 faces (99.945%). The remaining components were tiny fragments. The table, chair, crate, apple, walls, and floor were therefore not separate objects; they were fused into the main geometry.

`COLOR_0` was continuous appearance color with 35,862 unique RGB values, not a hidden class-ID encoding.

## Raycast and surface-query test

A 32 x 32 downward-ray grid produced:

- 1,024 rays
- 890 hits
- 86.9% hit rate
- Approximately 354 ms cold execution in Python with an R-tree

This confirms that collider raycasting and surface-height queries are practical. The mesh behaves like an inward-facing room shell, so triangle-surface/BVH collision is safer than treating it as a conventional solid volume.

## SPZ-to-collider alignment

The returned collider GLB and SPZ align best **without transforming either file** in their raw returned coordinate frames.

- Median mesh-to-splat nearest distance: approximately 0.32% of scene diagonal
- P90 distance: approximately 0.78% of scene diagonal

This is a strong geometric alignment result. For a metric scene, the same World Labs transform must be applied to both assets:

```text
metric_xyz = raw_xyz * 1.9479336738586426
metric_xyz.y -= 1.6518672704696655
```

Transforming only the SPZ breaks alignment. The raw ground plane for this world is near `y = 0.84801`.

One coordinate-system trap was observed: the exported PLY equals the decoded 100k SPZ only after flipping X and Z, a 180-degree rotation around Y. The transformed points then match exactly by index. The runtime must normalize asset coordinate conventions explicitly.

## Camera and semantic data

The API returned an equirectangular panorama, caption, thumbnail, Marble URL, metric scale, and ground offset. It did not return:

- camera intrinsics
- camera extrinsics or source camera trajectory
- per-point or per-triangle semantic labels
- object or instance IDs
- independent object transforms
- robot rig, joint state, or animation

Dataset cameras and their poses therefore must be created and recorded by our renderer.

## Product implication

Use World Labs for what it demonstrably provides:

1. Generate visually diverse spatial worlds.
2. Load the SPZ for appearance and the aligned collider for raycast/collision.
3. Derive static affordances such as floor and support surfaces.
4. Add the robot and manipulable props as separate assets with known IDs and transforms.
5. Record our own RGB/depth/camera pose/joint/action/object-state stream.
6. Optionally project 2D masks into the collider to create a semantic object graph for static scene elements.

The differentiated product is not “World Labs already gives robot data.” It is a **semantic world compiler that converts an unstructured World Labs world into a robot-ready dataset environment**.

## Audit artifacts and scripts

- `scripts/fetch_worldlabs_assets.mjs`
- `scripts/fetch_worldlabs_mesh_export.mjs`
- `scripts/inspect_worldlabs_assets.mjs`
- `scripts/inspect_spz.py`
- `scripts/check_worldlabs_alignment.py`

Raw generated assets and API responses are stored under `artifacts/worldlabs-audit/` and excluded from Git because responses contain expiring asset URLs.

## Official references

- [World API quickstart](https://docs.worldlabs.ai/api)
- [Get a world](https://docs.worldlabs.ai/api/reference/worlds/get)
- [Export a world](https://docs.worldlabs.ai/api/reference/worlds/export)
- [Rendering Marble SPZ](https://docs.worldlabs.ai/api/rendering-spz)
- [SPZ format specification](https://github.com/nianticlabs/spz/blob/main/README.md)
