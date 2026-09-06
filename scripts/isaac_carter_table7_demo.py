#!/usr/bin/env python3
"""Render the single World2Work hackathon demo shot in Isaac Sim.

The World Labs Marble collider is imported at its metric transform and kept in
the stage as geometry provenance.  Its matching panorama is used as the
readable visual environment.  An official NVIDIA Nova Carter asset is moved
kinematically along a full metric café aisle and stops at TABLE 7.  This is a
demo renderer only: it intentionally does not train or evaluate a policy.
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path


WIDTH = 1280
HEIGHT = 720
FPS = 12.0
START_HOLD_FRAMES = 12
MOVE_FRAMES = 78
SUCCESS_HOLD_FRAMES = 30
ROUTE_START_XY = (5.57645, -0.39966)
TABLE7_STOP_XY = (4.52645, 6.95034)
# Presentation proxy placed just beyond the reachable navigation goal.  Its
# dimensions below come from a segmented cafe table in the Marble geometry.
TABLE7_CENTER_XY = (5.38, 7.78)
TABLE7_SIZE_XY = (0.842, 0.972)
TABLE7_HEIGHT_M = 0.72


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panorama",
        type=Path,
        default=Path("/isaac-sim/worldlab/corgi_pano.png"),
    )
    parser.add_argument(
        "--collider-usd",
        type=Path,
        default=Path("/isaac-sim/output/corgi_collider_converted.usd"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/isaac-sim/output/world2work_carter_table7_demo.mp4"),
    )
    parser.add_argument(
        "--stage-output",
        type=Path,
        default=Path("/isaac-sim/output/world2work_carter_table7_demo.usda"),
    )
    parser.add_argument("--metric-scale", type=float, default=2.052333354949951)
    parser.add_argument("--ground-offset", type=float, default=1.671196460723877)
    parser.add_argument("--visual-z-bias", type=float, default=-0.32)
    parser.add_argument("--dome-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--preview-scale-sheet",
        action="store_true",
        help="Render a 3-up Carter scale comparison instead of an MP4.",
    )
    return parser.parse_args()


def bezier_pose(progress: float) -> tuple[tuple[float, float, float], float]:
    """Return a smooth metric aisle pose and heading for the demo route."""

    # This is the full 7.43 m collision-free aisle route, rather than a
    # compressed animation path.  One stage unit is exactly one metre.
    p0 = ROUTE_START_XY
    p1 = (5.55, 1.90)
    p2 = (4.75, 4.80)
    p3 = TABLE7_STOP_XY
    t = max(0.0, min(1.0, progress))
    u = 1.0 - t
    x = u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0]
    y = u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]
    dx = 3 * u * u * (p1[0] - p0[0]) + 6 * u * t * (p2[0] - p1[0]) + 3 * t * t * (p3[0] - p2[0])
    dy = 3 * u * u * (p1[1] - p0[1]) + 6 * u * t * (p2[1] - p1[1]) + 3 * t * t * (p3[1] - p2[1])
    return (x, y, 0.0), math.atan2(dy, dx)


def route_length_m(sample_count: int = 400) -> float:
    points = [bezier_pose(index / sample_count)[0] for index in range(sample_count + 1)]
    return sum(math.dist(a[:2], b[:2]) for a, b in zip(points, points[1:]))


def follow_camera_pose(position, yaw: float):
    """Return a shoulder-follow camera in the same metric coordinate frame."""

    distance = 2.75
    # Camera sits to Carter's left, keeping the right-side TABLE 7 readable.
    lateral = 1.05
    height = 1.52
    look_ahead = 0.72
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    camera_position = (
        position[0] - distance * cos_yaw - lateral * sin_yaw,
        position[1] - distance * sin_yaw + lateral * cos_yaw,
        position[2] + height,
    )
    look_at = (
        position[0] + look_ahead * cos_yaw,
        position[1] + look_ahead * sin_yaw,
        position[2] + 0.42,
    )
    return camera_position, look_at


def add_cube(stage, path: str, position, size, color):
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xform = UsdGeom.XformCommonAPI(cube)
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetScale(Gf.Vec3f(*size))
    return cube


def add_route_visuals(stage) -> None:
    from pxr import Gf, UsdGeom

    # Small metric breadcrumbs make the motion legible without covering the
    # World Labs floor with an artificial simulator grid.
    for index in range(11):
        position, _ = bezier_pose(index / 10.0)
        dot = UsdGeom.Cylinder.Define(stage, f"/World/Route/Dot_{index:02d}")
        dot.CreateAxisAttr("Z")
        dot.CreateRadiusAttr(0.020)
        dot.CreateHeightAttr(0.012)
        dot.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.42, 0.03)])
        dot.CreateDisplayOpacityAttr([0.72])
        UsdGeom.XformCommonAPI(dot).SetTranslate(
            Gf.Vec3d(position[0], position[1], 0.014)
        )

    goal = UsdGeom.Cylinder.Define(stage, "/World/Route/Table7Goal")
    goal.CreateAxisAttr("Z")
    goal.CreateRadiusAttr(0.30)
    goal.CreateHeightAttr(0.014)
    goal.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.92, 0.48)])
    goal.CreateDisplayOpacityAttr([0.30])
    UsdGeom.XformCommonAPI(goal).SetTranslate(
        Gf.Vec3d(TABLE7_STOP_XY[0], TABLE7_STOP_XY[1], 0.016)
    )


def add_metric_proxy_scene(stage) -> None:
    """Create a depth-bearing metric foreground in front of the panorama.

    A lat-long panorama has no floor depth or occlusion, so it cannot establish
    the apparent size of a robot.  This real 1 m/unit floor fixes that while the
    World Labs panorama remains the surrounding visual context.
    """

    from pxr import Gf, UsdGeom

    # Robust collider floor bounds (1st-99th percentile) from the Marble mesh.
    floor_center = (0.89068, 3.40078, -0.03)
    floor_size = (9.87846, 14.25089, 0.05)
    add_cube(
        stage,
        "/World/MetricForeground/FloorBase",
        floor_center,
        floor_size,
        (0.19, 0.085, 0.030),
    )

    # Alternating 0.44 m boards give the eye a real, repeated metric cue as the
    # follow camera moves.  They are visual-only and sit just above the base.
    nominal_board_length = 0.44
    y_min = floor_center[1] - floor_size[1] / 2.0
    board_count = int(math.ceil(floor_size[1] / nominal_board_length))
    board_length = floor_size[1] / board_count
    board_colors = (
        (0.43, 0.235, 0.095),
        (0.50, 0.285, 0.120),
        (0.37, 0.185, 0.068),
    )
    for index in range(board_count):
        board_y = y_min + (index + 0.5) * board_length
        add_cube(
            stage,
            f"/World/MetricForeground/Boards/Board_{index:02d}",
            (floor_center[0], board_y, -0.0025),
            (floor_size[0] - 0.02, board_length - 0.014, 0.005),
            board_colors[index % len(board_colors)],
        )

    # Sparse longitudinal joints prevent the floor from reading as a flat set
    # of stripes without turning it into a simulator grid.
    for index, x_value in enumerate((-2.44, -0.22, 2.00, 4.22)):
        seam = add_cube(
            stage,
            f"/World/MetricForeground/LongJoint_{index}",
            (x_value, floor_center[1], 0.0015),
            (0.012, floor_size[1] - 0.03, 0.003),
            (0.12, 0.050, 0.018),
        )
        seam.CreateDisplayOpacityAttr([0.55])


def add_table7_visual(stage) -> None:
    """Add a clearly readable, visual-only café table beside the stop pad.

    No Physics or Collision API is applied here.  The table is presentation
    geometry and therefore cannot interfere with the Carter motion renderer.
    """

    from pxr import Gf, UsdGeom

    table_x, table_y = TABLE7_CENTER_XY
    orange = (0.96, 0.36, 0.055)
    dark_wood = (0.24, 0.105, 0.035)
    green = (0.08, 0.92, 0.48)

    UsdGeom.Xform.Define(stage, "/World/Table7Visual")

    table_width, table_depth = TABLE7_SIZE_XY
    top_thickness = 0.05
    top_z = TABLE7_HEIGHT_M - top_thickness / 2.0

    # Metadata-grounded cafe table: 0.842 x 0.972 x 0.7025 m.  It sits just
    # beyond the route endpoint so Carter visibly parks without intersecting it.
    add_cube(
        stage,
        "/World/Table7Visual/Top",
        (table_x, table_y, top_z),
        (table_width, table_depth, top_thickness),
        orange,
    )
    for index, (offset_x, offset_y) in enumerate(
        (
            (-0.32, -0.385),
            (-0.32, 0.385),
            (0.32, -0.385),
            (0.32, 0.385),
        )
    ):
        add_cube(
            stage,
            f"/World/Table7Visual/Leg_{index}",
            (table_x + offset_x, table_y + offset_y, (TABLE7_HEIGHT_M - top_thickness) / 2.0),
            (0.045, 0.045, TABLE7_HEIGHT_M - top_thickness),
            dark_wood,
        )

    # Front-facing placard plus a simple geometric numeral 7.  This stays
    # legible from the fixed aisle camera and matches the green goal marker.
    add_cube(
        stage,
        "/World/Table7Visual/NumberPlate",
        (table_x, table_y - table_depth / 2.0 - 0.018, 0.855),
        (0.24, 0.025, 0.17),
        (0.035, 0.055, 0.075),
    )
    add_cube(
        stage,
        "/World/Table7Visual/Number7Top",
        (table_x, table_y - table_depth / 2.0 - 0.034, 0.900),
        (0.105, 0.012, 0.018),
        green,
    )
    seven_stem = add_cube(
        stage,
        "/World/Table7Visual/Number7Stem",
        (table_x + 0.018, table_y - table_depth / 2.0 - 0.034, 0.840),
        (0.020, 0.012, 0.105),
        green,
    )
    UsdGeom.XformCommonAPI(seven_stem).SetRotate(Gf.Vec3f(0.0, -24.0, 0.0))

    shadow = UsdGeom.Cylinder.Define(stage, "/World/Table7Visual/Shadow")
    shadow.CreateAxisAttr("Z")
    shadow.CreateRadiusAttr(0.52)
    shadow.CreateHeightAttr(0.006)
    shadow.CreateDisplayColorAttr([Gf.Vec3f(0.015, 0.02, 0.025)])
    shadow.CreateDisplayOpacityAttr([0.34])
    shadow_xform = UsdGeom.XformCommonAPI(shadow)
    shadow_xform.SetTranslate(Gf.Vec3d(table_x, table_y, 0.005))
    shadow_xform.SetScale(Gf.Vec3f(table_width / table_depth, 1.0, 1.0))


def add_service_kit(stage) -> None:
    """Turn the Carter mobile base into a readable coffee-serving robot."""

    from pxr import Gf, UsdGeom

    # A contact shadow card visually grounds Carter on the panorama floor.
    shadow = UsdGeom.Cylinder.Define(stage, "/World/CarterMotion/ServiceKit/Shadow")
    shadow.CreateAxisAttr("Z")
    shadow.CreateRadiusAttr(0.34)
    shadow.CreateHeightAttr(0.006)
    shadow.CreateDisplayColorAttr([Gf.Vec3f(0.015, 0.02, 0.025)])
    shadow.CreateDisplayOpacityAttr([0.42])
    shadow_xform = UsdGeom.XformCommonAPI(shadow)
    shadow_xform.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.006))
    shadow_xform.SetScale(Gf.Vec3f(1.0, 0.72, 1.0))

    # Keep the payload compact so the official Carter silhouette stays
    # recognizable.  The earlier tall mast made the base read as a 1 m-wide
    # serving cart in the depthless panorama.
    add_cube(
        stage,
        "/World/CarterMotion/ServiceKit/Tray",
        (0.0, 0.0, 0.405),
        (0.36, 0.30, 0.025),
        (0.96, 0.36, 0.055),
    )
    cup = UsdGeom.Cylinder.Define(stage, "/World/CarterMotion/ServiceKit/Coffee")
    cup.CreateAxisAttr("Z")
    cup.CreateRadiusAttr(0.038)
    cup.CreateHeightAttr(0.10)
    cup.CreateDisplayColorAttr([Gf.Vec3f(0.96, 0.95, 0.90)])
    UsdGeom.XformCommonAPI(cup).SetTranslate(Gf.Vec3d(0.02, 0.0, 0.468))


def alpha_box(cv2, frame, top_left, bottom_right, color, alpha: float):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def put_label(cv2, frame, text, origin, scale, color, thickness=1):
    # A thin black stroke keeps the HUD readable on both the windows and wood.
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, text, origin, font, scale, (8, 12, 18), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def decorate_frame(cv2, frame, index: int) -> None:
    move_start = START_HOLD_FRAMES
    move_end = START_HOLD_FRAMES + MOVE_FRAMES
    if index < move_start:
        phase = "SCENE READY"
        progress = 0.0
        phase_color = (82, 201, 255)
    elif index < move_end:
        phase = "RUN  /  NAVIGATING TO TABLE 7"
        progress = (index - move_start) / max(1, MOVE_FRAMES - 1)
        phase_color = (82, 201, 255)
    else:
        phase = "SUCCESS  /  TABLE 7 REACHED"
        progress = 1.0
        phase_color = (112, 242, 132)

    alpha_box(cv2, frame, (24, 22), (1256, 91), (12, 17, 24), 0.72)
    put_label(cv2, frame, "WORLD LABS MARBLE 3D", (48, 56), 0.72, (255, 255, 255), 1)
    put_label(cv2, frame, "ISAAC SIM  /  NOVA CARTER", (850, 56), 0.56, (220, 231, 239), 1)
    put_label(cv2, frame, "CORGI CAFE  ->  TABLE 7", (48, 82), 0.43, (122, 190, 255), 1)

    alpha_box(cv2, frame, (24, 640), (1256, 698), (10, 15, 20), 0.76)
    put_label(cv2, frame, phase, (48, 676), 0.62, phase_color, 1)
    bar_x0, bar_x1, bar_y0, bar_y1 = 790, 1225, 659, 676
    cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x1, bar_y1), (67, 77, 87), 1)
    fill_x = int(bar_x0 + (bar_x1 - bar_x0) * progress)
    if fill_x > bar_x0:
        cv2.rectangle(frame, (bar_x0 + 2, bar_y0 + 2), (fill_x, bar_y1 - 2), phase_color, -1)

    if index < move_start:
        alpha_box(cv2, frame, (418, 286), (862, 416), (9, 14, 21), 0.70)
        put_label(cv2, frame, "METRIC SCALE VERIFIED", (475, 333), 0.75, (255, 255, 255), 1)
        put_label(cv2, frame, "MARBLE 2.052x  ->  ISAAC METERS", (461, 377), 0.55, (122, 190, 255), 1)
    elif index >= move_end:
        alpha_box(cv2, frame, (48, 120), (575, 244), (11, 78, 45), 0.83)
        put_label(cv2, frame, "SUCCESS", (82, 174), 1.10, (144, 255, 173), 2)
        put_label(cv2, frame, "NOVA CARTER ARRIVED AT TABLE 7", (82, 218), 0.58, (255, 255, 255), 1)


def frame_progress(index: int) -> float:
    if index < START_HOLD_FRAMES:
        return 0.0
    if index >= START_HOLD_FRAMES + MOVE_FRAMES:
        return 1.0
    linear = (index - START_HOLD_FRAMES) / max(1, MOVE_FRAMES - 1)
    return linear * linear * (3.0 - 2.0 * linear)


def main() -> None:
    args = parse_args()
    if not args.panorama.is_file():
        raise FileNotFoundError(f"World Labs panorama not found: {args.panorama}")
    if not args.collider_usd.is_file():
        raise FileNotFoundError(f"Converted World Labs collider not found: {args.collider_usd}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stage_output.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "width": WIDTH,
            "height": HEIGHT,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
            "sync_loads": True,
            "disable_viewport_updates": False,
        }
    )

    annotator = None
    render_product = None
    video_writer = None
    try:
        import cv2
        import numpy as np
        import omni.replicator.core as rep
        import omni.timeline
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

        stage_utils.create_new_stage()
        simulation_app.reset_render_settings()
        stage = stage_utils.get_current_stage()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.Xform.Define(stage, "/World")

        # Exact World Labs -> Isaac metric transform:
        # (x,y,z)_marble -> (s*x, s*z, ground_offset - s*y).
        cafe_root = UsdGeom.Xform.Define(stage, "/World/WorldLabsMarble")
        cafe_xform = UsdGeom.XformCommonAPI(cafe_root)
        cafe_xform.SetTranslate(
            Gf.Vec3d(0.0, 0.0, args.ground_offset + args.visual_z_bias)
        )
        cafe_xform.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0))
        cafe_xform.SetScale(
            Gf.Vec3f(args.metric_scale, args.metric_scale, args.metric_scale)
        )
        geometry = UsdGeom.Xform.Define(stage, "/World/WorldLabsMarble/Geometry")
        geometry.GetPrim().GetReferences().AddReference(str(args.collider_usd))

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac Sim asset root is unavailable")
        carter_asset = assets_root + "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
        motion = UsdGeom.Xform.Define(stage, "/World/CarterMotion")
        motion_xform = UsdGeom.XformCommonAPI(motion)
        start_position, start_yaw = bezier_pose(0.0)
        motion_xform.SetTranslate(Gf.Vec3d(*start_position))
        motion_xform.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(start_yaw)))
        motion_xform.SetScale(Gf.Vec3f(1.0, 1.0, 1.0))
        stage_utils.add_reference_to_stage(
            carter_asset,
            "/World/CarterMotion/NovaCarter",
            variants=[
                ("Physics", "physx"),
                ("Sensors", "None"),
                ("Configuration", "Base"),
                ("ROS", "Disabled"),
            ],
        )
        add_service_kit(stage)
        add_metric_proxy_scene(stage)
        add_route_visuals(stage)
        add_table7_visual(stage)

        dome = UsdLux.DomeLight.Define(stage, "/World/WorldLabsPanorama")
        dome.CreateIntensityAttr(900.0)
        dome.CreateExposureAttr(1.0)
        dome.CreateTextureFileAttr(Sdf.AssetPath(str(args.panorama)))
        dome.CreateTextureFormatAttr("latlong")
        UsdGeom.XformCommonAPI(dome).SetRotate(
            Gf.Vec3f(0.0, 0.0, args.dome_yaw_deg)
        )
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(2500.0)
        key.CreateAngleAttr(1.0)
        UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(-35.0, 25.0, 20.0))

        carter_prim_count = 0
        for frame in range(600):
            simulation_app.update()
            if frame % 10 == 0:
                carter_prim_count = sum(
                    1
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith("/World/CarterMotion/NovaCarter")
                )
                if frame >= 45 and carter_prim_count >= 5:
                    break
        if carter_prim_count < 5:
            raise RuntimeError(f"Nova Carter reference did not resolve: {carter_asset}")

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        bbox_range = bbox_cache.ComputeWorldBound(motion.GetPrim()).ComputeAlignedRange()
        bbox_size = bbox_range.GetSize()
        carter_extent_m = [float(bbox_size[i]) for i in range(3)]
        # The service tray intentionally makes Z ~0.9 m. Horizontal size is the
        # useful sanity check: anything many metres wide would mean a unit bug.
        if not (0.35 <= max(carter_extent_m[0], carter_extent_m[1]) <= 1.25):
            raise RuntimeError(f"Nova Carter metric extent is implausible: {carter_extent_m}")

        # Preserve the properly scaled collider in the USD proof artifact, then
        # hide its fused/untextured visualization for the readable pano render.
        UsdGeom.Imageable(cafe_root.GetPrim()).MakeInvisible()
        stage.GetRootLayer().Export(str(args.stage_output))

        omni.timeline.get_timeline_interface().pause()
        rep.orchestrator.set_capture_on_play(False)
        initial_camera_position, initial_look_at = follow_camera_pose(
            start_position, start_yaw
        )
        camera = rep.functional.create.camera(
            position=initial_camera_position,
            look_at=initial_look_at,
            look_at_up_axis=(0.0, 0.0, 1.0),
            focal_length=27.0,
            clipping_range=(0.05, 100.0),
            parent="/World",
            name="World2WorkCarterCamera",
        )
        render_product = rep.create.render_product(
            camera, (WIDTH, HEIGHT), name="World2WorkCarterRenderProduct"
        )
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        annotator.attach(render_product)

        if args.preview_scale_sheet:
            tiles = []
            preview_position, preview_yaw = bezier_pose(0.62)
            for label, scale_value in (("A", 0.65), ("B", 0.82), ("C", 1.00)):
                motion_xform.SetTranslate(Gf.Vec3d(*preview_position))
                motion_xform.SetRotate(
                    Gf.Vec3f(0.0, 0.0, math.degrees(preview_yaw))
                )
                motion_xform.SetScale(
                    Gf.Vec3f(scale_value, scale_value, scale_value)
                )
                preview_camera_position, preview_look_at = follow_camera_pose(
                    preview_position, preview_yaw
                )
                rep.functional.modify.pose(
                    camera,
                    position_value=preview_camera_position,
                    look_at_value=preview_look_at,
                    look_at_up_axis=(0.0, 0.0, 1.0),
                    write_to_usd=True,
                )
                simulation_app.update()
                rep.orchestrator.step(
                    rt_subframes=8,
                    delta_time=0.0,
                    pause_timeline=True,
                    wait_for_render=True,
                )
                payload = annotator.get_data()
                if isinstance(payload, dict):
                    payload = payload.get("data")
                if payload is None:
                    raise RuntimeError(f"No RGB data for scale preview {label}")
                rgb = np.asarray(payload)[:, :, :3]
                if rgb.dtype != np.uint8:
                    maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
                    if maximum <= 1.0:
                        rgb = rgb * 255.0
                    rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
                bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                bgr = cv2.resize(bgr, (640, 360))
                alpha_box(cv2, bgr, (18, 16), (212, 72), (10, 15, 20), 0.80)
                put_label(
                    cv2,
                    bgr,
                    f"{label}  /  {scale_value:.2f}x",
                    (35, 53),
                    0.72,
                    (255, 255, 255),
                    1,
                )
                tiles.append(bgr)
            note = np.zeros((360, 640, 3), dtype=np.uint8)
            note[:] = (20, 25, 31)
            put_label(cv2, note, "CARTER SCALE CHECK", (48, 83), 0.90, (255, 255, 255), 1)
            put_label(cv2, note, "A  0.65x  /  compact", (48, 143), 0.62, (122, 190, 255), 1)
            put_label(cv2, note, "B  0.82x  /  balanced", (48, 193), 0.62, (122, 190, 255), 1)
            put_label(cv2, note, "C  1.00x  /  physical USD", (48, 243), 0.62, (122, 190, 255), 1)
            put_label(cv2, note, "PANO HAS NO DEPTH: EYE-MATCH DEMO", (48, 309), 0.48, (160, 171, 181), 1)
            sheet = np.vstack((np.hstack((tiles[0], tiles[1])), np.hstack((tiles[2], note))))
            preview_path = args.output.with_suffix(".scale-options.png")
            cv2.imwrite(str(preview_path), sheet)
            print(
                "CARTER_SCALE_SHEET_OK "
                + json.dumps({"output": str(preview_path), "scales": [0.65, 0.82, 1.0]}),
                flush=True,
            )
            return

        codec = "mp4v"
        video_writer = cv2.VideoWriter(
            str(args.output),
            cv2.VideoWriter_fourcc(*codec),
            FPS,
            (WIDTH, HEIGHT),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video output: {args.output}")

        total_frames = START_HOLD_FRAMES + MOVE_FRAMES + SUCCESS_HOLD_FRAMES
        still_indices = {0: "start", START_HOLD_FRAMES + MOVE_FRAMES // 2: "mid", total_frames - 1: "success"}
        captured = 0
        for frame_index in range(total_frames):
            progress = frame_progress(frame_index)
            position, yaw = bezier_pose(progress)
            motion_xform.SetTranslate(Gf.Vec3d(*position))
            motion_xform.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))
            camera_position, camera_look_at = follow_camera_pose(position, yaw)
            rep.functional.modify.pose(
                camera,
                position_value=camera_position,
                look_at_value=camera_look_at,
                look_at_up_axis=(0.0, 0.0, 1.0),
                write_to_usd=True,
            )
            simulation_app.update()
            rep.orchestrator.step(
                rt_subframes=4,
                delta_time=0.0,
                pause_timeline=True,
                wait_for_render=True,
            )
            payload = annotator.get_data()
            if isinstance(payload, dict):
                payload = payload.get("data")
            if payload is None:
                raise RuntimeError(f"RGB annotator returned no data at frame {frame_index}")
            rgb = np.asarray(payload)[:, :, :3]
            if rgb.dtype != np.uint8:
                maximum = float(np.nanmax(rgb)) if rgb.size else 0.0
                if maximum <= 1.0:
                    rgb = rgb * 255.0
                rgb = np.clip(rgb, 0.0, 255.0).astype(np.uint8)
            if rgb.shape[:2] != (HEIGHT, WIDTH):
                rgb = cv2.resize(rgb, (WIDTH, HEIGHT))
            bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
            decorate_frame(cv2, bgr, frame_index)
            video_writer.write(bgr)
            captured += 1
            still_name = still_indices.get(frame_index)
            if still_name:
                cv2.imwrite(str(args.output.with_name(f"{args.output.stem}_{still_name}.png")), bgr)
            if frame_index % 24 == 0 or frame_index == total_frames - 1:
                print(f"RENDER_PROGRESS {frame_index + 1}/{total_frames}", flush=True)

        video_writer.release()
        video_writer = None
        rep.orchestrator.wait_until_complete()
        result = {
            "output": str(args.output),
            "stage": str(args.stage_output),
            "frames": captured,
            "fps": FPS,
            "duration_s": captured / FPS,
            "resolution": [WIDTH, HEIGHT],
            "codec": codec,
            "robot_asset": carter_asset,
            "robot_extent_m": carter_extent_m,
            "marble_metric_scale": args.metric_scale,
            "marble_ground_offset_m": args.ground_offset,
            "robot_wrapper_scale": [1.0, 1.0, 1.0],
            "route_length_m": route_length_m(),
            "stop_distance_to_table_m": math.hypot(
                TABLE7_CENTER_XY[0] - TABLE7_STOP_XY[0],
                TABLE7_CENTER_XY[1] - TABLE7_STOP_XY[1],
            ),
            "stop_distance_to_table_edge_m": math.hypot(
                max(
                    abs(TABLE7_STOP_XY[0] - TABLE7_CENTER_XY[0])
                    - TABLE7_SIZE_XY[0] / 2.0,
                    0.0,
                ),
                max(
                    abs(TABLE7_STOP_XY[1] - TABLE7_CENTER_XY[1])
                    - TABLE7_SIZE_XY[1] / 2.0,
                    0.0,
                ),
            ),
            "table7_visual_center_xy_m": list(TABLE7_CENTER_XY),
            "table7_visual_size_m": [*TABLE7_SIZE_XY, TABLE7_HEIGHT_M],
            "table7_visual_physics": "visual_only_no_collision",
            "table7_provenance_note": "presentation proxy; dimensions grounded in a Marble-segmented cafe table",
            "metric_foreground_floor_center_m": [0.89068, 3.40078, -0.03],
            "metric_foreground_floor_size_m": [9.87846, 14.25089, 0.05],
            "success": captured == total_frames and args.output.stat().st_size > 1024,
        }
        args.output.with_suffix(".json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print("CARTER_TABLE7_VIDEO_OK " + json.dumps(result, sort_keys=True), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if video_writer is not None:
            video_writer.release()
        if annotator is not None:
            try:
                annotator.detach()
            except Exception:
                pass
        if render_product is not None:
            try:
                render_product.destroy()
            except Exception:
                pass
        simulation_app.close(wait_for_replicator=True)


if __name__ == "__main__":
    main()
