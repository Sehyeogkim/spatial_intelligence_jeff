#!/usr/bin/env python3
"""Segment a fused World Labs Marble collider into floor / walls / table tops.

World Labs returns one unlabeled, fused collider mesh (Y-up, non-metric).  This
script is the post-processing step that sits between the Marble download and
the Isaac Sim load:

  collider.glb  ->  segment_marble_collider.py  ->  segmented.glb (+ per-class
                                                      GLBs, semantics.json,
                                                      segmentation.png)

Classification is purely geometric.  Every triangle is transformed into the
verified Isaac Z-up metric frame, then labeled by its face normal and height:

  floor       horizontal, near the estimated floor level
  ceiling     horizontal, near the top of the shell
  seat        horizontal, chair-seat height band
  table_top   horizontal, table-height band (instanced: table_1, table_2, ...)
  wall        vertical, reaches above --wall-min-z
  furniture   vertical below wall height (table legs, chair backs, counters)
  other       sloped faces (cushions, lamp shades, mesh noise)

Table-top instances are clustered on a 2D grid so that each physical table
becomes its own named node in the output GLB.  Node names survive the Isaac
asset converter, so `/World/MarbleCafe/Geometry/table_3` can receive its own
semantic label and collision settings downstream.

By default the geometry is written back in the *raw Marble frame* so the
existing Isaac loading transform (rotate -90 X, scale, ground offset) keeps
working unchanged.  Pass --frame metric to write Z-up metric coordinates,
which is what you want for Blender inspection.

Only numpy is required.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marble_to_grid import (  # noqa: E402
    _decode_accessor,
    _mat_identity,
    _mat_mul,
    _node_matrix,
    _read_glb,
    load_world_scale,
    write_rgb_png,
)

CLASS_ORDER = ["floor", "ceiling", "wall", "furniture", "seat", "table_top", "other", "outside"]
CLASS_COLORS = {
    "floor": (200, 200, 200),
    "ceiling": (245, 245, 245),
    "wall": (90, 90, 110),
    "furniture": (170, 120, 60),
    "seat": (60, 160, 220),
    "table_top": (230, 70, 70),
    "other": (140, 190, 90),
    "outside": (225, 225, 240),
    "empty": (255, 255, 255),
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_mesh_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return raw (scene-space) vertices (N,3) and faces (M,3) for the GLB."""
    gltf, binary = _read_glb(path)
    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    roots = scene.get("nodes", list(range(len(gltf.get("nodes", [])))))
    vertex_chunks: list[np.ndarray] = []
    face_chunks: list[np.ndarray] = []
    offset = 0

    def visit(node_index: int, parent) -> None:
        nonlocal offset
        node = gltf["nodes"][node_index]
        world = _mat_mul(parent, _node_matrix(node))
        if "mesh" in node:
            mesh = gltf["meshes"][node["mesh"]]
            for primitive in mesh.get("primitives", []):
                attributes = primitive.get("attributes", {})
                if primitive.get("mode", 4) != 4 or "POSITION" not in attributes:
                    continue
                positions = np.asarray(_decode_accessor(gltf, binary, attributes["POSITION"]), dtype=np.float64)
                matrix = np.asarray(world, dtype=np.float64).reshape(4, 4).T  # column-major -> row-major
                if not np.allclose(matrix, np.eye(4)):
                    rotation, translation = matrix[:3, :3], matrix[:3, 3]
                    positions = np.einsum("ij,nj->ni", rotation, positions) + translation
                if "indices" in primitive:
                    indices = np.asarray(_decode_accessor(gltf, binary, primitive["indices"]), dtype=np.int64)
                else:
                    indices = np.arange(len(positions), dtype=np.int64)
                indices = indices[: len(indices) - len(indices) % 3].reshape(-1, 3)
                vertex_chunks.append(positions)
                face_chunks.append(indices + offset)
                offset += len(positions)
        for child in node.get("children", []):
            visit(child, world)

    for root in roots:
        visit(root, _mat_identity())
    if not face_chunks:
        raise ValueError("GLB contains no triangle primitives")
    vertices = np.concatenate(vertex_chunks, axis=0)
    faces = np.concatenate(face_chunks, axis=0)
    info = {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(faces)),
        "node_count": len(gltf.get("nodes", [])),
        "mesh_count": len(gltf.get("meshes", [])),
    }
    return vertices, faces, info


def raw_to_metric(raw: np.ndarray, scale: float, ground_offset: float) -> np.ndarray:
    """Verified Marble Y-up -> Isaac Z-up: (x,y,z) -> (s*x, s*z, g - s*y)."""
    out = np.empty_like(raw)
    out[:, 0] = scale * raw[:, 0]
    out[:, 1] = scale * raw[:, 2]
    out[:, 2] = ground_offset - scale * raw[:, 1]
    return out


def resolve_transform(args: argparse.Namespace) -> tuple[float, float, str]:
    if args.scale is not None and args.ground_offset is not None:
        return float(args.scale), float(args.ground_offset), "cli"
    if args.grid_meta and Path(args.grid_meta).is_file():
        meta = json.loads(Path(args.grid_meta).read_text(encoding="utf-8"))
        transform = meta.get("transform", {})
        if "scale" in transform and "ground_offset" in transform:
            return float(transform["scale"]), float(transform["ground_offset"]), args.grid_meta
    if args.world and Path(args.world).is_file():
        scale, ground = load_world_scale(Path(args.world))
        return scale, ground, args.world
    raise SystemExit("cannot resolve the metric transform: pass --scale/--ground-offset, --grid-meta or --world")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def face_geometry(metric: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b, c = metric[faces[:, 0]], metric[faces[:, 1]], metric[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    safe = np.where(area > 0, 2.0 * area, 1.0)
    normals = cross / safe[:, None]
    centroids = (a + b + c) / 3.0
    return normals, area, centroids


def weighted_height_mode(z: np.ndarray, weight: np.ndarray, lo: float, hi: float, bin_m: float = 0.02) -> float | None:
    mask = (z >= lo) & (z <= hi) & (weight > 0)
    if not mask.any():
        return None
    edges = np.arange(lo, hi + bin_m, bin_m)
    hist, _ = np.histogram(z[mask], bins=edges, weights=weight[mask])
    best = int(np.argmax(hist))
    if hist[best] <= 0:
        return None
    inside = mask & (z >= edges[best]) & (z < edges[best + 1])
    return float(np.average(z[inside], weights=weight[inside]))


def classify(normals: np.ndarray, area: np.ndarray, centroids: np.ndarray, params: dict) -> tuple[np.ndarray, dict]:
    z = centroids[:, 2]
    nz = np.abs(normals[:, 2])
    horizontal = nz >= math.cos(math.radians(params["horizontal_deg"]))
    vertical = nz <= math.sin(math.radians(params["vertical_deg"]))

    floor_z = params["floor_z"]
    if floor_z is None:
        floor_z = weighted_height_mode(z, area * horizontal, -0.6, 0.6)
        if floor_z is None:
            floor_z = 0.0
    ceiling_z = params["ceiling_z"]
    if ceiling_z is None:
        top = float(np.quantile(z, 0.995))
        ceiling_z = weighted_height_mode(z, area * horizontal, max(floor_z + 1.8, top - 1.0), top + 0.05)
        if ceiling_z is None:
            ceiling_z = top

    rel = z - floor_z
    labels = np.full(len(z), CLASS_ORDER.index("other"), dtype=np.int8)
    tol = params["floor_tolerance"]
    labels[horizontal & (np.abs(rel) <= tol)] = CLASS_ORDER.index("floor")
    labels[horizontal & (z >= ceiling_z - params["ceiling_tolerance"])] = CLASS_ORDER.index("ceiling")
    seat_lo, seat_hi = params["seat_band"]
    labels[horizontal & (rel > seat_lo) & (rel <= seat_hi)] = CLASS_ORDER.index("seat")
    table_lo, table_hi = params["table_band"]
    labels[horizontal & (rel > table_lo) & (rel <= table_hi)] = CLASS_ORDER.index("table_top")
    labels[vertical & (rel >= params["wall_min_z"])] = CLASS_ORDER.index("wall")
    labels[vertical & (rel < params["wall_min_z"])] = CLASS_ORDER.index("furniture")
    crop = params.get("crop_xy")
    outside_count = 0
    if crop:
        xmin, xmax, ymin, ymax = crop
        outside = (centroids[:, 0] < xmin) | (centroids[:, 0] > xmax) | (centroids[:, 1] < ymin) | (centroids[:, 1] > ymax)
        labels[outside] = CLASS_ORDER.index("outside")
        outside_count = int(outside.sum())

    # Vertical faces that belong to a wall stretch continuously from low to high.
    # Promote "furniture" faces whose XY column also carries wall faces above,
    # so wall bottoms are not mislabeled as furniture.
    wall_idx = CLASS_ORDER.index("wall")
    furniture_idx = CLASS_ORDER.index("furniture")
    cell = params["cluster_cell"]
    keys = np.floor(centroids[:, :2] / cell).astype(np.int64)
    wall_cells = set(map(tuple, keys[labels == wall_idx]))
    candidate = np.where(labels == furniture_idx)[0]
    promote = [i for i in candidate if tuple(keys[i]) in wall_cells and rel[i] >= params["wall_promote_min_z"]]
    labels[promote] = wall_idx

    return labels, {"floor_z": float(floor_z), "ceiling_z": float(ceiling_z), "promoted_wall_faces": len(promote), "outside_crop_faces": outside_count}


# --------------------------------------------------------------------------- #
# Instancing
# --------------------------------------------------------------------------- #


def cluster_xy(indices: np.ndarray, centroids: np.ndarray, cell: float, gap_cells: int = 1) -> list[np.ndarray]:
    """Group triangles by 2D grid connectivity (8-neighbourhood, optional gap)."""
    if len(indices) == 0:
        return []
    keys = np.floor(centroids[indices, :2] / cell).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for tri, key in zip(indices, map(tuple, keys)):
        buckets.setdefault(key, []).append(int(tri))
    seen: set[tuple[int, int]] = set()
    clusters: list[np.ndarray] = []
    reach = range(-gap_cells, gap_cells + 1)
    for start in buckets:
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        members: list[int] = []
        while queue:
            key = queue.popleft()
            members.extend(buckets[key])
            for dx in reach:
                for dy in reach:
                    neighbour = (key[0] + dx, key[1] + dy)
                    if neighbour in buckets and neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
        clusters.append(np.asarray(members, dtype=np.int64))
    return clusters


def build_instances(labels: np.ndarray, class_name: str, prefix: str, area: np.ndarray, centroids: np.ndarray,
                    metric: np.ndarray, faces: np.ndarray, params: dict) -> tuple[list[dict], np.ndarray]:
    class_idx = CLASS_ORDER.index(class_name)
    indices = np.where(labels == class_idx)[0]
    instance_of = np.full(len(labels), -1, dtype=np.int64)
    clusters = cluster_xy(indices, centroids, params["cluster_cell"], params["cluster_gap_cells"])
    records = []
    for members in clusters:
        total = float(area[members].sum())
        if total < params["min_instance_area"]:
            continue
        verts = metric[np.unique(faces[members])]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        weights = area[members]
        centre = np.average(centroids[members], axis=0, weights=weights)
        records.append({
            "triangles": members,
            "area_m2": total,
            "centroid_xyz_m": [round(float(v), 4) for v in centre],
            "height_z_m": round(float(np.average(centroids[members, 2], weights=weights)), 4),
            "bbox_min_m": [round(float(v), 4) for v in lo],
            "bbox_max_m": [round(float(v), 4) for v in hi],
            "footprint_xy_m": [round(float(hi[0] - lo[0]), 3), round(float(hi[1] - lo[1]), 3)],
        })
    records.sort(key=lambda r: -r["area_m2"])
    for number, record in enumerate(records, start=1):
        record["name"] = f"{prefix}_{number}"
        instance_of[record["triangles"]] = number
    demoted = np.where((labels == class_idx) & (instance_of < 0))[0]
    labels[demoted] = CLASS_ORDER.index("other")  # tiny fragments are not real surfaces
    return records, instance_of


# --------------------------------------------------------------------------- #
# GLB writing
# --------------------------------------------------------------------------- #


def _pad(data: bytes, alignment: int = 4, fill: bytes = b"\x00") -> bytes:
    remainder = len(data) % alignment
    return data if remainder == 0 else data + fill * (alignment - remainder)


def write_glb(path: Path, groups: list[dict], vertices: np.ndarray, faces: np.ndarray, asset_extras: dict) -> dict:
    """Write one GLB with one named node/mesh per group.

    groups: [{"name": str, "triangles": np.ndarray, "extras": dict}, ...]
    vertices: (N,3) positions in the desired output frame.
    """
    binary = bytearray()
    buffer_views = []
    accessors = []
    meshes = []
    nodes = []
    summary = {}
    for group in groups:
        tri = faces[group["triangles"]]
        unique, remapped = np.unique(tri.reshape(-1), return_inverse=True)
        positions = vertices[unique].astype("<f4")
        indices = remapped.astype("<u4").reshape(-1)
        pos_bytes = positions.tobytes()
        idx_bytes = indices.tobytes()

        pos_view = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(pos_bytes), "target": 34962})
        binary += _pad(pos_bytes)
        idx_view = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(idx_bytes), "target": 34963})
        binary += _pad(idx_bytes)

        pos_accessor = len(accessors)
        accessors.append({
            "bufferView": pos_view, "componentType": 5126, "count": int(len(positions)), "type": "VEC3",
            "min": [float(v) for v in positions.min(axis=0)], "max": [float(v) for v in positions.max(axis=0)],
        })
        idx_accessor = len(accessors)
        accessors.append({"bufferView": idx_view, "componentType": 5125, "count": int(len(indices)), "type": "SCALAR"})

        meshes.append({
            "name": group["name"],
            "primitives": [{"attributes": {"POSITION": pos_accessor}, "indices": idx_accessor, "mode": 4}],
        })
        nodes.append({"name": group["name"], "mesh": len(meshes) - 1, "extras": group.get("extras", {})})
        summary[group["name"]] = {"vertices": int(len(positions)), "triangles": int(len(tri))}

    gltf = {
        "asset": {"version": "2.0", "generator": "segment_marble_collider.py", "extras": asset_extras},
        "scene": 0,
        "scenes": [{"name": "segmented_collider", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(binary)}],
    }
    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), fill=b" ")
    bin_chunk = _pad(bytes(binary))
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    with path.open("wb") as handle:
        handle.write(b"glTF" + struct.pack("<II", 2, total))
        handle.write(struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk)
        handle.write(struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk)
    return summary


# --------------------------------------------------------------------------- #
# Preview
# --------------------------------------------------------------------------- #


def write_preview(path: Path, labels: np.ndarray, metric: np.ndarray, faces: np.ndarray, cell: float, instance_of: np.ndarray) -> None:
    xy = metric[:, :2]
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    cols = int(math.ceil((hi[0] - lo[0]) / cell)) + 1
    rows = int(math.ceil((hi[1] - lo[1]) / cell)) + 1
    if rows * cols > 4_000_000:
        raise ValueError("preview raster too large; increase --preview-cell")
    priority = {name: CLASS_ORDER.index(name) for name in CLASS_ORDER}
    paint_order = ["ceiling", "outside", "other", "floor", "wall", "furniture", "seat", "table_top"]
    ranking = np.full((rows, cols), -1, dtype=np.int64)
    pixel = np.full((rows, cols, 3), 255, dtype=np.uint8)
    ceiling = priority["ceiling"]
    for rank, name in enumerate(paint_order):
        idx = np.where(labels == priority[name])[0]
        if len(idx) == 0:
            continue
        if name == "ceiling":
            continue  # a ceiling covers everything in top-down view; leave it out
        # sample vertices + centroids so thin surfaces still show
        p0, p1, p2 = metric[faces[idx, 0], :2], metric[faces[idx, 1], :2], metric[faces[idx, 2], :2]
        points = np.concatenate([p0, p1, p2, (p0 + p1 + p2) / 3.0, (p0 + p1) / 2, (p1 + p2) / 2, (p0 + p2) / 2])
        cx = np.clip(((points[:, 0] - lo[0]) / cell).astype(int), 0, cols - 1)
        cy = np.clip(((points[:, 1] - lo[1]) / cell).astype(int), 0, rows - 1)
        mask = ranking[cy, cx] < rank
        colour = np.array(CLASS_COLORS[name], dtype=np.uint8)
        if name == "table_top":
            inst = np.tile(instance_of[idx], 7)
            for cy_i, cx_i, m, k in zip(cy, cx, mask, inst):
                if m:
                    shade = 40 * (k % 5)
                    pixel[cy_i, cx_i] = (min(255, 150 + shade), 60, 60 + shade)
                    ranking[cy_i, cx_i] = rank
            continue
        pixel[cy[mask], cx[mask]] = colour
        ranking[cy[mask], cx[mask]] = rank
    _ = ceiling
    rows_out = [[tuple(int(c) for c in pixel[r, c]) for c in range(cols)] for r in range(rows)]
    write_rgb_png(path, rows_out, scale=4)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collider", default="artifacts/corgi-cafe/collider.glb")
    parser.add_argument("--world", default="artifacts/corgi-cafe/world.json")
    parser.add_argument("--grid-meta", default="artifacts/corgi-cafe/nav/grid_meta.json",
                        help="Reuse the transform (and table_7 goal) recorded by marble_to_grid.py")
    parser.add_argument("--out-dir", default="artifacts/corgi-cafe/semantic")
    parser.add_argument("--frame", choices=["raw", "metric"], default="raw",
                        help="Output coordinate frame. raw = unchanged Marble frame (drop-in for Isaac loader); metric = Isaac Z-up metres")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--ground-offset", type=float, default=None)
    parser.add_argument("--floor-z", type=float, default=None, help="Floor height in metric frame (auto: area-weighted mode)")
    parser.add_argument("--ceiling-z", type=float, default=None, help="Ceiling height in metric frame (auto)")
    parser.add_argument("--horizontal-deg", type=float, default=30.0, help="Max tilt from horizontal for floor/table faces")
    parser.add_argument("--vertical-deg", type=float, default=30.0, help="Max tilt from vertical for wall faces")
    parser.add_argument("--floor-tolerance", type=float, default=0.15)
    parser.add_argument("--ceiling-tolerance", type=float, default=0.30)
    parser.add_argument("--seat-band", type=float, nargs=2, default=(0.30, 0.55), metavar=("LO", "HI"))
    parser.add_argument("--table-band", type=float, nargs=2, default=(0.55, 1.10), metavar=("LO", "HI"))
    parser.add_argument("--wall-min-z", type=float, default=1.40, help="Vertical faces at/above this height are walls")
    parser.add_argument("--wall-promote-min-z", type=float, default=0.0,
                        help="Vertical faces below wall height are still walls if wall faces sit in the same XY column above")
    parser.add_argument("--cluster-cell", type=float, default=0.10, help="XY cell (m) used to cluster table-top instances")
    parser.add_argument("--cluster-gap-cells", type=int, default=1)
    parser.add_argument("--min-instance-area", type=float, default=0.25, help="Discard table/seat clusters smaller than this (m^2)")
    parser.add_argument("--preview-cell", type=float, default=0.05)
    parser.add_argument("--crop", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="Metric XY crop; faces outside become class 'outside'. Default: initial_crop_xy_m from --grid-meta")
    parser.add_argument("--no-crop", action="store_true", help="Disable the crop even when grid_meta provides one")
    args = parser.parse_args()

    collider = Path(args.collider)
    if not collider.is_file():
        raise SystemExit(f"collider not found: {collider}")
    scale, ground_offset, transform_source = resolve_transform(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, faces, info = load_mesh_arrays(collider)
    metric = raw_to_metric(raw, scale, ground_offset)
    normals, area, centroids = face_geometry(metric, faces)

    crop_xy = None if args.no_crop else args.crop
    if crop_xy is None and not args.no_crop and args.grid_meta and Path(args.grid_meta).is_file():
        meta = json.loads(Path(args.grid_meta).read_text(encoding="utf-8"))
        crop_xy = meta.get("extraction", {}).get("initial_crop_xy_m")
    params = {
        "crop_xy": [float(v) for v in crop_xy] if crop_xy else None,
        "horizontal_deg": args.horizontal_deg,
        "vertical_deg": args.vertical_deg,
        "floor_z": args.floor_z,
        "ceiling_z": args.ceiling_z,
        "floor_tolerance": args.floor_tolerance,
        "ceiling_tolerance": args.ceiling_tolerance,
        "seat_band": list(args.seat_band),
        "table_band": list(args.table_band),
        "wall_min_z": args.wall_min_z,
        "wall_promote_min_z": args.wall_promote_min_z,
        "cluster_cell": args.cluster_cell,
        "cluster_gap_cells": args.cluster_gap_cells,
        "min_instance_area": args.min_instance_area,
    }
    labels, levels = classify(normals, area, centroids, params)
    tables, table_instance = build_instances(labels, "table_top", "table", area, centroids, metric, faces, params)
    seats, _ = build_instances(labels, "seat", "seat", area, centroids, metric, faces, params)

    # Optional: relate detected tables to the navigation goal proxy from grid_meta.
    goal_match = None
    if args.grid_meta and Path(args.grid_meta).is_file() and tables:
        meta = json.loads(Path(args.grid_meta).read_text(encoding="utf-8"))
        target = meta.get("targets", {}).get("table_7")
        if target and "xy_m" in target:
            gx, gy = target["xy_m"]
            distances = [math.hypot(t["centroid_xyz_m"][0] - gx, t["centroid_xyz_m"][1] - gy) for t in tables]
            best = int(np.argmin(distances))
            goal_match = {
                "grid_meta_table_7_xy_m": [gx, gy],
                "nearest_detected_table": tables[best]["name"],
                "distance_m": round(distances[best], 3),
                "note": "table_7 in grid_meta is a free-space proxy; the nearest detected table_top is the physical candidate",
            }

    # Output geometry frame.
    out_vertices = raw if args.frame == "raw" else metric
    asset_extras = {
        "source_collider": str(collider),
        "frame": args.frame,
        "metric_transform": {"formula": "raw (x,y,z) -> (s*x, s*z, ground_offset - s*y)", "scale": scale, "ground_offset": ground_offset},
    }

    groups = []
    class_summary = {}
    for name in CLASS_ORDER:
        idx = np.where(labels == CLASS_ORDER.index(name))[0]
        class_summary[name] = {"triangles": int(len(idx)), "area_m2": round(float(area[idx].sum()), 3)}
        if name == "table_top":
            for table in tables:
                groups.append({"name": table["name"], "triangles": table["triangles"],
                               "extras": {"class": "table_top", "instance": table["name"], "height_z_m": table["height_z_m"],
                                          "centroid_xyz_m": table["centroid_xyz_m"], "frame_of_extras": "metric"}})
            continue
        if len(idx) == 0:
            continue
        groups.append({"name": name, "triangles": idx, "extras": {"class": name}})

    combined = out_dir / "segmented.glb"
    node_summary = write_glb(combined, groups, out_vertices, faces, asset_extras)
    per_class_files = {}
    for name in CLASS_ORDER:
        members = [g for g in groups if g["extras"]["class"] == name]
        if not members:
            continue
        target = out_dir / f"{name}.glb"
        write_glb(target, members, out_vertices, faces, asset_extras)
        per_class_files[name] = str(target)

    np.save(out_dir / "triangle_labels.npy", labels)
    np.save(out_dir / "triangle_table_instance.npy", table_instance)
    preview = out_dir / "segmentation.png"
    write_preview(preview, labels, metric, faces, args.preview_cell, table_instance)

    def strip(record: dict) -> dict:
        return {k: v for k, v in record.items() if k != "triangles"} | {"triangles": int(len(record["triangles"]))}

    semantics = {
        "schema_version": "world2work.collider-semantics.v1",
        "source": {"collider": str(collider), **info, "transform_source": transform_source},
        "transform": asset_extras["metric_transform"],
        "output": {"frame": args.frame, "combined_glb": str(combined), "per_class_glb": per_class_files,
                   "triangle_labels_npy": str(out_dir / "triangle_labels.npy"), "preview_png": str(preview),
                   "label_index": {i: n for i, n in enumerate(CLASS_ORDER)}},
        "parameters": params,
        "levels": levels,
        "classes": class_summary,
        "nodes": node_summary,
        "tables": [strip(t) for t in tables],
        "seats": [strip(s) for s in seats],
        "goal_match": goal_match,
        "method": "geometric: face normal + height bands in the Isaac Z-up metric frame; table tops clustered on an XY grid",
        "caveats": [
            "Labels are heuristic; a counter or shelf at table height is reported as a table_top.",
            "The collider is an inward-facing fused shell, so table legs are labeled 'furniture', not attached to their table instance.",
            "Coordinates in this file are always metric Z-up regardless of --frame.",
            "Faces outside the XY crop are kept in the GLB as node 'outside' but never form table/seat instances.",
        ],
    }
    (out_dir / "semantics.json").write_text(json.dumps(semantics, indent=2), encoding="utf-8")

    print(json.dumps({
        "collider": str(collider),
        "triangles": info["triangle_count"],
        "floor_z": round(levels["floor_z"], 3),
        "ceiling_z": round(levels["ceiling_z"], 3),
        "crop_xy": params["crop_xy"],
        "classes": {k: v["triangles"] for k, v in class_summary.items()},
        "tables": [(t["name"], round(t["area_m2"], 2), t["centroid_xyz_m"]) for t in tables],
        "seats": len(seats),
        "goal_match": goal_match,
        "out_dir": str(out_dir),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
