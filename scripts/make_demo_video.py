#!/usr/bin/env python3
"""Assemble the 30-second World2Work demo video from real artifacts.

Segments (1280x720 @ 30 fps):
  A  0-3 s   source photo -> Marble panorama
  B  3-8 s   occupancy grid + value heatmap + learning curve timelapse
  C  8-26 s  untrained vs trained, same held-out start (Isaac frames if present,
             otherwise a grid-sim preview drawn from evaluation_summary.json)
  D 26-30 s  held-out scoreboard + tagline

Every number on screen is read from artifacts written by the training and
evaluation scripts; nothing is typed in by hand. Drop Isaac Sim PNG sequences in
assets/isaac_frames/trained and assets/isaac_frames/untrained to replace the
segment-C preview automatically.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1280, 720, 30
BG = (18, 20, 26)
INK = (240, 240, 236)
MUTED = (150, 154, 162)
ACCENT = (237, 125, 49)   # World2Work orange (matches value_heatmap target colour)
GREEN = (31, 170, 89)
RED = (220, 70, 60)
FREE = (52, 56, 66)
WALL = (28, 30, 36)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
                  "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize((math.ceil(image.width * scale), math.ceil(image.height * scale)), Image.LANCZOS)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def pano_view(pano: Image.Image, center_u: float = 0.5, fov: float = 0.42) -> Image.Image:
    """Crop an equirectangular panorama to a 16:9 window centred at horizontal fraction center_u."""
    crop_w = int(pano.width * fov)
    crop_h = int(crop_w * 9 / 16)
    left = int(pano.width * center_u - crop_w / 2)
    top = int(pano.height * 0.5 - crop_h * 0.55)
    left = max(0, min(pano.width - crop_w, left))
    return pano.crop((left, top, left + crop_w, top + crop_h)).resize((W, H), Image.LANCZOS)


def colour_map(t: float) -> tuple[int, int, int]:
    t = min(1.0, max(0.0, t))
    r, g, b = colorsys.hsv_to_rgb(0.64 * (1.0 - t), 0.78, 0.96)
    return int(255 * r), int(255 * g), int(255 * b)


def text(draw: ImageDraw.ImageDraw, xy, s: str, size: int, fill=INK, bold=False, anchor="la"):
    draw.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def bfs(goal, free):
    from collections import deque
    dist = {goal: 0}
    q = deque([goal])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            n = (r + dr, c + dc)
            if n in free and n not in dist:
                dist[n] = dist[(r, c)] + 1
                q.append(n)
    return dist


class Assets:
    def __init__(self, nav: Path, cafe: Path):
        self.grid = load_json(nav / "occupancy_grid.json")
        self.meta = load_json(nav / "grid_meta.json")
        self.policy = load_json(nav / "policy_table_7.json")
        self.curve = load_json(nav / "learning_curve.json")
        self.evaluation = load_json(nav / "evaluation_summary.json")
        self.route = load_json(nav / "route_table_7.json")
        self.rows, self.cols = self.grid["rows"], self.grid["cols"]
        self.free = {(r, c) for r in range(self.rows) for c in range(self.cols) if self.grid["cells"][r][c] == self.grid["free_value"]}
        self.goal = tuple(self.meta["targets"]["table_7"]["cell"])
        self.dist = bfs(self.goal, self.free)
        self.q = {tuple(int(v) for v in k.split(",")): vals for k, vals in self.policy["q_values"].items()}
        self.photo = Image.open(cafe / "source-google-maps.jpg").convert("RGB")
        self.pano = Image.open(cafe / "pano.png").convert("RGB")
        experience = nav / "training_experience.jsonl"
        self.transitions = sum(1 for _ in experience.open(encoding="utf-8")) if experience.exists() else None
        pols = self.evaluation["policies"]
        self.untrained_key = next(k for k in pols if "untrained" in k or "random" in k)
        self.trained_key = next(k for k in pols if k != self.untrained_key)

    def metrics(self, key: str) -> dict:
        eps = self.evaluation["policies"][key]["episodes"]
        successes = [e for e in eps if e["success"]]
        actions = sum(e["actual_steps"] for e in eps)
        collisions = sum(e.get("collision_actions", 0) for e in eps)
        eff = [e["path_efficiency"] for e in successes if e.get("path_efficiency") is not None]
        return {
            "n": len(eps),
            "success": len(successes) / len(eps),
            "collision": collisions / actions if actions else 0.0,
            "efficiency": sum(eff) / len(eff) if eff else 0.0,
        }

    def held_out_pair(self, index: int = 0):
        u = self.evaluation["policies"][self.untrained_key]["episodes"]
        t = self.evaluation["policies"][self.trained_key]["episodes"]
        # Pick the first held-out configuration where both policies started at the same cell.
        by_start = {tuple(e["start_cell"]): e for e in t}
        for e in u:
            mate = by_start.get(tuple(e["start_cell"]))
            if mate and index == 0:
                return e, mate
            if mate:
                index -= 1
        return u[0], t[0]


# --- map drawing -------------------------------------------------------------

def draw_map(assets: Assets, size_px: int, heat: float = 0.0, trail=None, robot=None, collision_flash=0.0, label_goal=True) -> Image.Image:
    rows, cols = assets.rows, assets.cols
    img = Image.new("RGB", (cols * size_px, rows * size_px), WALL)
    d = ImageDraw.Draw(img)
    values = [max(assets.q[s]) for s in assets.free]
    lo, hi = min(values), max(values)
    for (r, c) in assets.free:
        x0, y0 = c * size_px, (rows - 1 - r) * size_px
        colour = FREE
        if heat > 0:
            hc = colour_map((max(assets.q[(r, c)]) - lo) / max(1e-9, hi - lo))
            colour = tuple(int(a * (1 - heat * 0.85) + b * heat * 0.85) for a, b in zip(FREE, hc))
        d.rectangle((x0 + 1, y0 + 1, x0 + size_px - 1, y0 + size_px - 1), fill=colour)
    if trail and len(trail) > 1:
        pts = [((c + 0.5) * size_px, (rows - 0.5 - r) * size_px) for r, c in trail]
        d.line(pts, fill=(255, 255, 255), width=max(2, size_px // 8))
    gr, gc = assets.goal
    gx, gy = (gc + 0.5) * size_px, (rows - 0.5 - gr) * size_px
    rad = size_px * 0.42
    d.ellipse((gx - rad, gy - rad, gx + rad, gy + rad), outline=ACCENT, width=max(2, size_px // 8))
    if label_goal:
        text(d, (gx, gy), "7", int(size_px * 0.6), fill=ACCENT, bold=True, anchor="mm")
    if robot is not None:
        rr, rc, heading = robot
        rx, ry = (rc + 0.5) * size_px, (rows - 0.5 - rr) * size_px
        body = size_px * 0.36
        colour = RED if collision_flash > 0 else GREEN
        d.rounded_rectangle((rx - body, ry - body * 0.75, rx + body, ry + body * 0.75), radius=size_px * 0.12, fill=colour)
        hx, hy = rx + math.cos(heading) * body, ry - math.sin(heading) * body
        d.ellipse((hx - size_px * 0.1, hy - size_px * 0.1, hx + size_px * 0.1, hy + size_px * 0.1), fill=INK)
        # coffee cup on the tray
        d.ellipse((rx - size_px * 0.1, ry - size_px * 0.1, rx + size_px * 0.1, ry + size_px * 0.1), fill=(255, 235, 200), outline=(90, 60, 30))
    return img


def draw_curve(assets: Assets, width: int, height: int, progress: float) -> Image.Image:
    img = Image.new("RGB", (width, height), (24, 26, 33))
    d = ImageDraw.Draw(img)
    rolling = assets.curve["rolling"]
    n = max(2, int(len(rolling) * ease(progress)))
    pad = 44
    d.line((pad, height - pad, width - 16, height - pad), fill=MUTED, width=2)
    d.line((pad, 16, pad, height - pad), fill=MUTED, width=2)
    text(d, (pad - 6, 16), "100%", 14, fill=MUTED, anchor="ra")
    text(d, (pad - 6, height - pad), "0%", 14, fill=MUTED, anchor="rs")
    text(d, (width - 16, height - pad + 6), f"{rolling[-1]['episode']} episodes", 14, fill=MUTED, anchor="ra")
    pts = []
    rew = []
    for i, point in enumerate(rolling[:n]):
        x = pad + (width - pad - 16) * (point["episode"] / rolling[-1]["episode"])
        y = height - pad - (height - pad - 16) * point["success_rate_fixed_start"]
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill=GREEN, width=3)
        d.ellipse((pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5), fill=GREEN)
        label_anchor = "rt" if pts[-1][1] < 48 else "rb"
        label_y = pts[-1][1] + 10 if label_anchor == "rt" else pts[-1][1] - 8
        text(d, (pts[-1][0] - 8, label_y), f"{rolling[n - 1]['success_rate_fixed_start']:.0%}", 16, fill=GREEN, bold=True, anchor=label_anchor)
    text(d, (pad + 4, 14), "success rate (rolling 100 episodes)", 14, fill=MUTED)
    return img


# --- segments -----------------------------------------------------------------

def segment_a(assets: Assets, frames: list) -> None:
    photo = cover(assets.photo, W, H)
    pano = pano_view(assets.pano, center_u=0.26)
    for i in range(3 * FPS):
        t = i / (3 * FPS)
        if t < 0.45:
            base = photo.copy()
        else:
            k = ease((t - 0.45) / 0.35)
            base = Image.blend(photo, pano, k)
        d = ImageDraw.Draw(base)
        d.rectangle((0, H - 120, W, H), fill=(0, 0, 0))
        text(d, (40, H - 100), "Corgi Cafe, San Francisco", 34, bold=True)
        sub = "one photo" if t < 0.45 else "one photo  →  reconstructed 3D world  (World Labs Marble)"
        text(d, (40, H - 56), sub, 22, fill=MUTED)
        frames.append(base)


def segment_b(assets: Assets, frames: list) -> None:
    pano = pano_view(assets.pano, center_u=0.26).filter(ImageFilter.GaussianBlur(6))
    dark = Image.new("RGB", (W, H), BG)
    bg = Image.blend(pano, dark, 0.72)
    cell = 24
    n = 5 * FPS
    for i in range(n):
        t = i / n
        heat = ease((t - 0.15) / 0.35)
        frame = bg.copy()
        m = draw_map(assets, cell, heat=heat)
        frame.paste(m, (120, (H - m.height) // 2))
        d = ImageDraw.Draw(frame)
        text(d, (120 + m.width // 2, (H - m.height) // 2 - 14), "café floor map from the Marble collider", 16, fill=MUTED, anchor="mb")
        curve = draw_curve(assets, 640, 300, ease((t - 0.25) / 0.7))
        frame.paste(curve, (500, 120))
        transitions = f"{assets.transitions:,} transitions" if assets.transitions else "training experience log"
        text(d, (500, 60), "Robot learns to navigate this room", 34, bold=True)
        text(d, (500, 440), f"tabular Q-learning · {len(assets.curve['episodes']):,} episodes · {transitions}", 20, fill=MUTED)
        text(d, (500, 470), "experience generated in a simulator derived from the reconstructed world", 18, fill=MUTED)
        if heat > 0.5:
            text(d, (500, 520), "value map after training", 18, fill=ACCENT)
        frames.append(frame)


def extract_mp4_frames(video: Path, count: int) -> Path | None:
    """Explode an Isaac Sim MP4 (from --video-output) into JPEG frames next to the video."""
    if not video.is_file():
        return None
    out = video.with_suffix("") / "frames"
    out.mkdir(parents=True, exist_ok=True)
    if len(list(out.glob("*.jpg"))) < 10:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = "ffmpeg"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(video), "-q:v", "2", str(out / "%05d.jpg")], check=True)
    return out


def load_isaac_frames(folder: Path, count: int) -> list[Image.Image] | None:
    # Accept either a folder of PNG frames (Replicator BasicWriter nests rgb_*.png)
    # or an MP4 rendered by isaac_nav_episode.py --video-output, placed as
    # assets/isaac_holdout_<name>.mp4.
    mp4 = folder.parent.parent / f"isaac_holdout_{folder.name}.mp4"
    exploded = extract_mp4_frames(mp4, count)
    if exploded is not None:
        folder = exploded
    files = sorted(folder.rglob("*.png")) + sorted(folder.rglob("*.jpg"))
    if len(files) < 10:
        return None
    picked = [files[min(len(files) - 1, int(i * len(files) / count))] for i in range(count)]
    return [cover(Image.open(f).convert("RGB"), W // 2, H) for f in picked]


def segment_c(assets: Assets, frames: list, isaac_dir: Path) -> None:
    n = 18 * FPS
    untrained_ep, trained_ep = assets.held_out_pair()
    cell_m = assets.meta["cell_size_m"]
    steps_shown = max(len(trained_ep["visited_cells"]), 2)
    step_frames = n / (steps_shown + 3)  # leave a beat after arrival
    isaac = {name: load_isaac_frames(isaac_dir / name, n) for name in ("untrained", "trained")}
    halves = [("UNTRAINED  ·  random policy", untrained_ep, isaac["untrained"], RED),
              ("TRAINED  ·  Q-learning policy", trained_ep, isaac["trained"], GREEN)]
    for i in range(n):
        frame = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(frame)
        for side, (title, ep, isaac_frames, colour) in enumerate(halves):
            x0 = side * (W // 2)
            cells = ep["visited_cells"]
            k = min(len(cells) - 1, i / step_frames)
            idx = int(k)
            frac = k - idx
            cur = cells[idx]
            nxt = cells[min(len(cells) - 1, idx + 1)]
            r = cur[0] + (nxt[0] - cur[0]) * frac
            c = cur[1] + (nxt[1] - cur[1]) * frac
            if nxt != cur:
                heading = math.atan2(nxt[0] - cur[0], nxt[1] - cur[1])
            else:
                heading = math.pi / 2
            collision_now = 1.0 if (nxt == cur and idx < len(cells) - 1 and frac < 0.5) else 0.0
            reached = idx >= len(cells) - 1 and ep["success"]
            if isaac_frames:
                frame.paste(isaac_frames[i], (x0, 0))
            else:
                size = 24
                m = draw_map(assets, size, heat=0.0, trail=cells[: idx + 1], robot=(r, c, heading), collision_flash=collision_now)
                frame.paste(m, (x0 + (W // 2 - m.width) // 2, 84))
                text(d, (x0 + W // 4, H - 22), "grid-sim preview of the held-out episode", 14, fill=MUTED, anchor="mb")
            d.rectangle((x0, 0, x0 + W // 2, 70), fill=(0, 0, 0))
            text(d, (x0 + 24, 20), title, 24, fill=colour, bold=True)
            steps_so_far = min(idx, len(cells) - 1)
            collisions = sum(1 for j in range(1, steps_so_far + 1) if cells[j] == cells[j - 1])
            remaining = assets.dist.get((cur[0], cur[1]), 0) * cell_m
            hud = f"steps {steps_so_far:>3}   collisions {collisions:>2}   to table 7  {remaining:4.1f} m"
            text(d, (x0 + 24, 46), hud, 16, fill=INK)
            if reached:
                text(d, (x0 + W // 4, H // 2), "DELIVERED", 44, fill=GREEN, bold=True, anchor="mm")
                cx, cy = x0 + W // 4, H // 2 + 52
                d.line((cx - 18, cy, cx - 6, cy + 12, cx + 20, cy - 14), fill=GREEN, width=6, joint="curve")
            elif idx >= len(cells) - 1:
                text(d, (x0 + W // 4, H // 2), "not delivered", 36, fill=RED, bold=True, anchor="mm")
        d.line((W // 2, 0, W // 2, H), fill=(60, 60, 70), width=2)
        text(d, (W // 2, H - 4), f"same held-out start cell {tuple(trained_ep['start_cell'])} · heading {trained_ep.get('initial_orientation', '')}", 14, fill=MUTED, anchor="md")
        frames.append(frame)


def segment_d(assets: Assets, frames: list) -> None:
    u, t = assets.metrics(assets.untrained_key), assets.metrics(assets.trained_key)
    n = 4 * FPS
    for i in range(n):
        tt = i / n
        frame = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(frame)
        text(d, (W // 2, 70), f"Held-out test  ·  {t['n']} episodes per policy  ·  start cells & headings excluded from training", 24, fill=MUTED, anchor="mm")
        rows = [("Success rate", f"{u['success']:.0%}", f"{t['success']:.0%}"),
                ("Collision rate", f"{u['collision']:.0%}", f"{t['collision']:.0%}"),
                ("Path efficiency", f"{u['efficiency']:.2f}", f"{t['efficiency']:.2f}")]
        text(d, (640, 140), "untrained", 22, fill=RED, anchor="mm")
        text(d, (960, 140), "trained", 22, fill=GREEN, anchor="mm")
        for j, (label, a, b) in enumerate(rows):
            y = 220 + j * 110
            reveal = ease((tt - 0.1 - j * 0.12) / 0.25)
            text(d, (120, y), label, 34, bold=True)
            if reveal > 0:
                text(d, (640, y), a, 48, fill=RED, bold=True, anchor="mm")
                text(d, (800, y), "→", 40, fill=MUTED, anchor="mm")
                text(d, (960, y), b, 48, fill=GREEN, bold=True, anchor="mm")
        if tt > 0.72:
            k = ease((tt - 0.72) / 0.2)
            col = tuple(int(BG[z] * (1 - k) + ACCENT[z] * k) for z in range(3))
            text(d, (W // 2, H - 70), "Dead spatial data  →  Living worlds  →  Robot experience", 30, fill=col, bold=True, anchor="mm")
        frames.append(frame)


def encode(frames_dir: Path, output: Path) -> None:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y", "-framerate", str(FPS), "-i", str(frames_dir / "%05d.jpg"), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-crf", "19", "-movflags", "+faststart", str(output)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nav-dir", default="artifacts/corgi-cafe/nav")
    parser.add_argument("--cafe-dir", default="artifacts/corgi-cafe")
    parser.add_argument("--isaac-frames", default="assets/isaac_frames")
    parser.add_argument("--frames-dir", default=None, help="scratch directory for JPEG frames")
    parser.add_argument("--output", default="assets/demo_30s.mp4")
    parser.add_argument("--segments", default="abcd")
    args = parser.parse_args()

    assets = Assets(Path(args.nav_dir), Path(args.cafe_dir))
    frames: list[Image.Image] = []
    if "a" in args.segments:
        segment_a(assets, frames)
    if "b" in args.segments:
        segment_b(assets, frames)
    if "c" in args.segments:
        segment_c(assets, frames, Path(args.isaac_frames))
    if "d" in args.segments:
        segment_d(assets, frames)

    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(args.output).with_suffix("") / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.jpg"):
        old.unlink()
    for index, frame in enumerate(frames):
        frame.save(frames_dir / f"{index:05d}.jpg", quality=92)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encode(frames_dir, output)
    print(json.dumps({"status": "VIDEO_OK", "output": str(output), "frames": len(frames), "seconds": round(len(frames) / FPS, 2),
                      "isaac_frames_used": {k: (Path(args.isaac_frames) / k).exists() and len(list((Path(args.isaac_frames) / k).glob('*.png'))) >= 10 for k in ("untrained", "trained")},
                      "metrics": {"untrained": assets.metrics(assets.untrained_key), "trained": assets.metrics(assets.trained_key)}}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VIDEO_ERROR: {exc}", file=sys.stderr)
        raise
