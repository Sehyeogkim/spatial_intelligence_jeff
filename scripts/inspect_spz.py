#!/usr/bin/env python3

import json
import os
import sys

import numpy as np
import spz


def main(path: str) -> None:
    options = spz.UnpackOptions()
    # World Labs collider GLB and this generated SPZ align in the returned raw
    # RUB frame. Convert both together only when a target engine requires it.
    options.to_coord = spz.CoordinateSystem.RUB
    cloud = spz.load_spz(path, options)

    point_count = cloud.num_points
    degree = cloud.sh_degree
    sh_width = ((degree + 1) ** 2 - 1) * 3
    positions = np.asarray(cloud.positions).reshape(point_count, 3)
    scales = np.asarray(cloud.scales).reshape(point_count, 3)
    rotations = np.asarray(cloud.rotations).reshape(point_count, 4)
    alphas = np.asarray(cloud.alphas)
    colors = np.asarray(cloud.colors).reshape(point_count, 3)
    sh = np.asarray(cloud.sh).reshape(point_count, sh_width)

    with np.errstate(over="ignore", invalid="ignore"):
        opacity = np.where(
            alphas >= 0,
            1.0 / (1.0 + np.exp(-alphas)),
            np.exp(alphas) / (1.0 + np.exp(alphas)),
        )

    result = {
        "file": os.path.basename(path),
        "file_bytes": os.path.getsize(path),
        "coordinate_system": "RUB",
        "num_points": point_count,
        "sh_degree": degree,
        "antialiased": bool(cloud.antialiased),
        "arrays": {
            "positions": list(positions.shape),
            "scales_log": list(scales.shape),
            "rotations_xyzw": list(rotations.shape),
            "alphas_logits": list(alphas.shape),
            "colors_base_rgb": list(colors.shape),
            "spherical_harmonics": list(sh.shape),
        },
        "bbox_min": positions.min(axis=0).tolist(),
        "bbox_max": positions.max(axis=0).tolist(),
        "bbox_size": np.ptp(positions, axis=0).tolist(),
        "log_scale_percentiles": np.percentile(scales, [0, 50, 95, 100]).tolist(),
        "opacity_percentiles": np.percentile(opacity, [0, 50, 95, 100]).tolist(),
        "quaternion_norm_percentiles": np.percentile(
            np.linalg.norm(rotations, axis=1), [0, 50, 100]
        ).tolist(),
        "color_range": [float(colors.min()), float(colors.max())],
        "contains_nan": {
            "positions": bool(np.isnan(positions).any()),
            "scales": bool(np.isnan(scales).any()),
            "rotations": bool(np.isnan(rotations).any()),
            "alphas": bool(np.isnan(alphas).any()),
            "colors": bool(np.isnan(colors).any()),
            "sh": bool(np.isnan(sh).any()),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FILE.spz")
    main(sys.argv[1])
