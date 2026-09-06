#!/usr/bin/env python3

import json
import os
import sys

import numpy as np
import spz
import trimesh
from scipy.spatial import cKDTree


def load_mesh(path: str):
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_mesh"):
            return loaded.to_mesh()
        return trimesh.util.concatenate(tuple(loaded.dump()))
    return loaded


def robust_bounds(points):
    low, high = np.quantile(points, [0.01, 0.99], axis=0)
    return {"q01": low.tolist(), "q99": high.tolist(), "size": (high - low).tolist()}


def main(root: str) -> None:
    world_response = json.load(open(os.path.join(root, "world.json"), encoding="utf8"))
    world = world_response.get("world", world_response)
    semantics = world["assets"]["splats"]["semantics_metadata"]
    scale = float(semantics["metric_scale_factor"])
    ground = float(semantics["ground_plane_offset"])

    cloud = spz.load_spz(os.path.join(root, "splats-500k.spz"))
    splats = np.asarray(cloud.positions).reshape(cloud.num_points, 3).astype(np.float64)
    mesh = load_mesh(os.path.join(root, "collider.glb"))
    mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)

    rng = np.random.default_rng(20260905)
    splat_sample = splats[rng.choice(len(splats), min(75000, len(splats)), replace=False)]
    mesh_surface, _ = trimesh.sample.sample_surface(mesh, min(30000, max(5000, len(mesh.faces) // 4)), seed=20260905)
    diagonal = float(np.linalg.norm(np.ptp(mesh_surface, axis=0)))

    metric_ground = splat_sample * scale
    metric_ground[:, 1] -= ground
    x180 = splat_sample * np.array([1.0, -1.0, -1.0])
    metric_ground_x180 = metric_ground * np.array([1.0, -1.0, -1.0])
    x180_metric_minus_y = x180 * scale
    x180_metric_minus_y[:, 1] -= ground
    x180_metric_plus_y = x180 * scale
    x180_metric_plus_y[:, 1] += ground
    candidates = {
        "spz_raw": splat_sample,
        "spz_x180": x180,
        "spz_metric_ground": metric_ground,
        "spz_metric_ground_then_x180": metric_ground_x180,
        "spz_x180_metric_minus_y": x180_metric_minus_y,
        "spz_x180_metric_plus_y": x180_metric_plus_y,
    }

    scores = []
    for name, points in candidates.items():
        distances, _ = cKDTree(points).query(mesh_surface, k=1, workers=-1)
        scores.append({
            "candidate": name,
            "mesh_to_splat_distance": {
                "median": float(np.median(distances)),
                "p90": float(np.percentile(distances, 90)),
                "p99": float(np.percentile(distances, 99)),
                "median_fraction_of_mesh_diagonal": float(np.median(distances) / diagonal),
                "p90_fraction_of_mesh_diagonal": float(np.percentile(distances, 90) / diagonal),
            },
            "splat_robust_bounds": robust_bounds(points),
        })
    scores.sort(key=lambda item: item["mesh_to_splat_distance"]["median_fraction_of_mesh_diagonal"])

    result = {
        "metric_scale_factor": scale,
        "ground_plane_offset": ground,
        "spz_points": int(cloud.num_points),
        "collider_vertices": int(len(mesh.vertices)),
        "collider_triangles": int(len(mesh.faces)),
        "collider_bounds": np.asarray(mesh.bounds).tolist(),
        "collider_robust_bounds": robust_bounds(mesh_vertices),
        "mesh_diagonal": diagonal,
        "ranking": scores,
        "interpretation": "Lower normalized nearest-neighbor distance is better. Treat <0.03 plus visual overlay as aligned; >0.05 suggests a scale/axis mismatch.",
    }
    output = os.path.join(root, "alignment.json")
    with open(output, "w", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/worldlabs-audit")
