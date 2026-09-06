#!/usr/bin/env python3
"""Headless frame capture for the Nova Carter café episode in Isaac Sim 6.0.

Reuses the panorama-dome + Replicator BasicWriter recipe that already produced
artifacts/isaac-smoke/corgi_world_franka_hero.png on the pod, but captures a
PNG per call so a driving episode becomes an image sequence.

Camera placement: the World Labs panorama was captured at the Marble frame
origin, which the grid_meta transform maps to (0, 0, ground_offset) in the
Isaac Z-up frame. A camera at exactly that point sees the panorama dome with
zero parallax for any yaw/pitch, so a robot driving on the metric floor plane
looks correctly embedded in the café. Do not move the camera off that point.

Integration into isaac_nav_episode.py (after the stage is built, before the
drive loop):

    from isaac_frame_capture import FrameCapture
    capture = FrameCapture(stage, out_dir=Path("/workspace/output/frames/trained"),
                           panorama=Path("/workspace/artifacts/corgi-cafe/pano.png"),
                           ground_offset=meta.ground_offset, look_at=(4.7, 3.3, 0.35))
    capture.setup(simulation_app)
    ...
    # inside the drive loop, after world.step(render=True):
    if step_index % capture.every == 0:
        capture.capture()
    ...
    capture.finish()   # prints FRAME_CAPTURE_OK with the frame count

Run the episode with --render and --disable-marble-visual (the fused collider
is untextured; the panorama supplies the visuals).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


class FrameCapture:
    def __init__(
        self,
        stage,
        out_dir: Path,
        panorama: Path,
        ground_offset: float,
        look_at: Sequence[float] = (4.7, 3.3, 0.35),
        resolution: Sequence[int] = (1280, 720),
        focal_length: float = 24.0,
        every: int = 3,
        rt_subframes: int = 4,
        dome_intensity: float = 900.0,
    ) -> None:
        self.stage = stage
        self.out_dir = Path(out_dir)
        self.panorama = Path(panorama)
        self.camera_position = (0.0, 0.0, float(ground_offset))
        self.look_at = tuple(float(v) for v in look_at)
        self.resolution = tuple(int(v) for v in resolution)
        self.focal_length = focal_length
        self.every = every
        self.rt_subframes = rt_subframes
        self.dome_intensity = dome_intensity
        self.frames = 0
        self._writer = None

    def setup(self, simulation_app) -> None:
        import omni.replicator.core as rep
        from pxr import Sdf, UsdLux

        dome = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome.CreateIntensityAttr(self.dome_intensity)
        dome.CreateExposureAttr(1.0)
        if self.panorama.is_file():
            dome.CreateTextureFileAttr(Sdf.AssetPath(str(self.panorama)))
            dome.CreateTextureFormatAttr("latlong")
        else:
            print(f"FRAME_CAPTURE_WARN panorama missing: {self.panorama}", flush=True)

        rep.orchestrator.set_capture_on_play(False)
        camera = rep.functional.create.camera(
            position=self.camera_position,
            look_at=self.look_at,
            focal_length=self.focal_length,
            clipping_range=(0.05, 100.0),
            parent="/World",
            name="EpisodeCamera",
        )
        render_product = rep.create.render_product(camera, self.resolution, name="EpisodeRenderProduct")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=str(self.out_dir))
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        writer.attach(render_product)
        self._writer = writer
        for _ in range(8):
            simulation_app.update()
        (self.out_dir / "camera.json").write_text(json.dumps({
            "position": self.camera_position,
            "look_at": self.look_at,
            "focal_length_mm": self.focal_length,
            "resolution": self.resolution,
            "yaw_deg": round(math.degrees(math.atan2(self.look_at[1] - self.camera_position[1], self.look_at[0] - self.camera_position[0])), 2),
            "note": "camera sits at the panorama capture point so the dome has no parallax",
        }, indent=2) + "\n", encoding="utf-8")
        print(f"FRAME_CAPTURE_READY out={self.out_dir} every={self.every} steps", flush=True)

    def capture(self) -> None:
        import omni.replicator.core as rep

        rep.orchestrator.step(rt_subframes=self.rt_subframes, delta_time=0.0)
        self.frames += 1

    def finish(self) -> int:
        import omni.replicator.core as rep

        rep.orchestrator.wait_until_complete()
        if self._writer is not None:
            self._writer.detach()
        written = sorted(self.out_dir.rglob("rgb_*.png"))  # BasicWriter may nest a subfolder
        print(f"FRAME_CAPTURE_OK frames={len(written)} dir={self.out_dir}", flush=True)
        return len(written)
