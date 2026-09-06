#!/usr/bin/env python3
"""Render one headless Isaac Sim frame with a World Labs collider and Franka.

Run this as the only Isaac/Kit process in the container. The input World Labs
GLB is converted to USD once and cached next to the requested output image.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import traceback
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collider",
        type=Path,
        default=Path("/workspace/artifacts/corgi-cafe/collider.glb"),
    )
    parser.add_argument(
        "--world-json",
        type=Path,
        default=Path("/workspace/artifacts/corgi-cafe/world.json"),
    )
    parser.add_argument(
        "--panorama",
        type=Path,
        default=Path("/workspace/artifacts/corgi-cafe/pano.png"),
    )
    parser.add_argument("--converted-usd", type=Path, default=None)
    parser.add_argument("--metric-scale", type=float, default=None)
    parser.add_argument("--ground-offset", type=float, default=None)
    parser.add_argument("--visual-z-bias", type=float, default=-0.32)
    parser.add_argument("--enable-collider", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/output/isaac_world_robot_smoke.png"),
    )
    return parser.parse_args()


def read_world_transform(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    world = payload.get("world", payload)
    metadata = world["assets"]["splats"]["semantics_metadata"]
    return (
        float(metadata["metric_scale_factor"]),
        float(metadata["ground_plane_offset"]),
    )


def main() -> None:
    args = parse_args()
    if not args.collider.is_file():
        raise FileNotFoundError(f"World Labs collider not found: {args.collider}")
    if not args.world_json.is_file() and (
        args.metric_scale is None or args.ground_offset is None
    ):
        raise FileNotFoundError(
            "World metadata is missing; pass both --metric-scale and "
            f"--ground-offset: {args.world_json}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # SimulationApp must be created before importing Omniverse/Isaac modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": True,
            "width": 1280,
            "height": 720,
            "renderer": "RaytracedLighting",
            "multi_gpu": False,
            "sync_loads": True,
            "disable_viewport_updates": False,
        }
    )

    writer = None
    render_product = None
    try:
        import omni.replicator.core as rep
        from isaacsim.core.experimental.utils import stage as stage_utils
        from isaacsim.core.experimental.utils.app import enable_extension
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics

        enable_extension("omni.kit.asset_converter")
        for _ in range(3):
            simulation_app.update()
        import omni.kit.asset_converter

        stage_utils.create_new_stage()
        simulation_app.reset_render_settings()
        stage = stage_utils.get_current_stage()
        UsdGeom.Xform.Define(stage, "/World")

        source_stat = args.collider.stat()
        converted_usd = args.converted_usd or args.output.parent / (
            f"corgi-collider-{source_stat.st_size}-{source_stat.st_mtime_ns}.usd"
        )

        async def convert_collider() -> None:
            context = omni.kit.asset_converter.AssetConverterContext()
            context.ignore_materials = False
            context.ignore_animation = True
            context.ignore_cameras = True
            context.single_mesh = True
            context.use_meter_as_world_unit = False
            context.create_world_as_default_root_prim = True

            task = omni.kit.asset_converter.get_instance().create_converter_task(
                str(args.collider),
                str(converted_usd),
                lambda _progress, _total: None,
                context,
            )
            if not await task.wait_until_finished():
                raise RuntimeError(
                    "GLB-to-USD conversion failed: "
                    f"{task.get_status()} {task.get_error_message()}"
                )

        if not converted_usd.is_file():
            simulation_app.run_coroutine(convert_collider())

        if args.world_json.is_file():
            metric_scale, ground_offset = read_world_transform(args.world_json)
        else:
            metric_scale = args.metric_scale
            ground_offset = args.ground_offset

        # World Labs collider coordinates are transformed into Isaac's Z-up frame:
        # (x, y, z)_raw -> (s*x, s*z, ground_offset - s*y).
        cafe_root = UsdGeom.Xform.Define(stage, "/World/CafeCollider")
        cafe_xform = UsdGeom.XformCommonAPI(cafe_root)
        cafe_xform.SetTranslate(
            Gf.Vec3d(0.0, 0.0, ground_offset + args.visual_z_bias)
        )
        cafe_xform.SetRotate(Gf.Vec3f(-90.0, 0.0, 0.0))
        cafe_xform.SetScale(Gf.Vec3f(metric_scale, metric_scale, metric_scale))
        # Reference directly so Isaac's metrics assembler does not inject its
        # glTF centimeter/Y-up compatibility ops. World Labs supplies its own
        # metric scale and ground transform, which we apply on CafeCollider.
        cafe_geometry = UsdGeom.Xform.Define(
            stage, "/World/CafeCollider/Geometry"
        )
        cafe_geometry.GetPrim().GetReferences().AddReference(str(converted_usd))

        assets_root = get_assets_root_path()
        if not assets_root:
            raise RuntimeError("Isaac Sim asset root is unavailable")
        robot_root = UsdGeom.Xform.Define(stage, "/World/RobotRoot")
        UsdGeom.XformCommonAPI(robot_root).SetTranslate(Gf.Vec3d(0.0, 0.8, 0.0))
        stage_utils.add_reference_to_stage(
            usd_path=(
                assets_root
                + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
            ),
            path="/World/RobotRoot/Franka",
            variants=[("Mesh", "Performance")],
        )

        # Mark the converted mesh as collision geometry as well as visible geometry.
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/CafeCollider") and prim.IsA(
                UsdGeom.Mesh
            ):
                mesh = UsdGeom.Mesh(prim)
                if args.enable_collider:
                    UsdPhysics.CollisionAPI.Apply(prim)
                mesh.CreateDoubleSidedAttr(True)

        # A tiny task island makes the robot/cup relationship legible while the
        # World Labs collider remains the imported scene geometry.
        table = UsdGeom.Cube.Define(stage, "/World/TaskIsland/Table")
        table.CreateSizeAttr(1.0)
        table.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.12, 0.04)])
        table_xform = UsdGeom.XformCommonAPI(table)
        table_xform.SetTranslate(Gf.Vec3d(0.0, 1.75, 0.72))
        table_xform.SetScale(Gf.Vec3f(1.25, 0.62, 0.10))
        cup = UsdGeom.Cylinder.Define(stage, "/World/TaskIsland/CoffeeCup")
        cup.CreateRadiusAttr(0.055)
        cup.CreateHeightAttr(0.14)
        cup.CreateAxisAttr("Z")
        cup.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.95, 0.92)])
        UsdGeom.XformCommonAPI(cup).SetTranslate(Gf.Vec3d(0.38, 1.50, 0.84))

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(900.0)
        dome.CreateExposureAttr(1.0)
        if args.panorama.is_file():
            dome.CreateTextureFileAttr(Sdf.AssetPath(str(args.panorama)))
            dome.CreateTextureFormatAttr("latlong")
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr(2500.0)
        key.CreateAngleAttr(1.0)
        UsdGeom.XformCommonAPI(key).SetRotate(Gf.Vec3f(-35.0, 25.0, 20.0))

        # Give both references time to resolve. Remote Isaac assets can take a
        # few seconds even when the local shader cache is already warm.
        cafe_mesh_count = 0
        franka_prim_count = 0
        for frame in range(600):
            simulation_app.update()
            if frame % 10 == 0:
                cafe_mesh_count = sum(
                    1
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith("/World/CafeCollider")
                    and prim.IsA(UsdGeom.Mesh)
                )
                franka_prim_count = sum(
                    1
                    for prim in stage.Traverse()
                    if str(prim.GetPath()).startswith("/World/RobotRoot/Franka")
                )
                if frame >= 45 and cafe_mesh_count >= 1 and franka_prim_count >= 2:
                    break
        print(
            f"IMPORT_CHECK cafe_meshes={cafe_mesh_count} "
            f"franka_prims={franka_prim_count}",
            flush=True,
        )
        if cafe_mesh_count < 1:
            raise RuntimeError("Converted World Labs USD contains no mesh prim")
        if franka_prim_count < 2:
            raise RuntimeError("Franka reference did not resolve")

        stage_output = args.output.with_suffix(".usda")
        stage.GetRootLayer().Export(str(stage_output))

        # The collider is fused and untextured. It is preserved in the USD stage
        # for geometry/collision work, while the matching World Labs panorama is
        # used for the readable hero frame.
        UsdGeom.Imageable(cafe_root.GetPrim()).MakeInvisible()

        rep.orchestrator.set_capture_on_play(False)
        camera = rep.functional.create.camera(
            position=(2.6, -3.4, 2.1),
            look_at=(0.0, 1.05, 0.78),
            focal_length=28.0,
            clipping_range=(0.05, 100.0),
            parent="/World",
            name="SmokeCamera",
        )
        render_product = rep.create.render_product(
            camera,
            (1280, 720),
            name="WorldRobotSmokeRenderProduct",
        )
        capture_dir = Path(
            tempfile.mkdtemp(
                prefix="isaac_world_robot_smoke_",
                dir=args.output.parent,
            )
        )
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(capture_dir))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach(render_product)

        for _ in range(8):
            simulation_app.update()
        rep.orchestrator.step(rt_subframes=16, delta_time=0.0)
        rep.orchestrator.wait_until_complete()

        captures = sorted(capture_dir.rglob("*.png"))
        if not captures:
            raise RuntimeError(f"Replicator produced no PNG under {capture_dir}")
        shutil.copy2(captures[-1], args.output)
        print(
            "ISAAC_WORLD_ROBOT_SMOKE_OK "
            + json.dumps(
                {
                    "output": str(args.output),
                    "bytes": args.output.stat().st_size,
                    "collider_usd": str(converted_usd),
                    "stage": str(stage_output),
                    "cafe_mesh_prims": cafe_mesh_count,
                    "franka_prims": franka_prim_count,
                    "metric_scale_factor": metric_scale,
                    "ground_plane_offset": ground_offset,
                    "visual_z_bias": args.visual_z_bias,
                    "collider_enabled": args.enable_collider,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.detach()
        if render_product is not None:
            render_product.destroy()
        simulation_app.close(wait_for_replicator=True)


if __name__ == "__main__":
    main()
