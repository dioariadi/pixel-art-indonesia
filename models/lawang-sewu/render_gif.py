"""Turntable GIF renderer (copied from ../gedung-sate/render_gif.py, defaults set for Lawang Sewu: 60 slices of 121x121, turn offset 45 deg so frame 0 faces the corner entrance, check at 0/45/90/135/180/270 deg) for a sprite-stack slice strip (ground slice on the left).

Per frame, bottom to top: rotate each slice by the turn angle about its symmetric centre
((w-1)/2, (h-1)/2) with nearest sampling, squash y for pitch, lift 1 px per slice.
A ground shadow (stack projected along a fixed sun vector) is drawn first; a 1 px dark
outline is traced once around the finished stack. Everything happens at 1x art pixels,
then the frame is upscaled by an integer factor with NEAREST.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
BG = (206, 226, 242)
SHADOW = (176, 194, 212)
INK = (24, 28, 48)


def load_strip(path: Path, count: int | None):
    a = np.array(Image.open(path).convert("RGBA"))
    h, total_w = a.shape[:2]
    if count is None:
        # square-ish slices are not guaranteed, so require the count from the caller or infer from h
        raise SystemExit("pass --count")
    assert total_w % count == 0, f"strip width {total_w} is not a multiple of {count}"
    w = total_w // count
    return np.stack([a[:, i * w : (i + 1) * w] for i in range(count)])  # [z, h, w, 4]


def render(vol: np.ndarray, turn: float, squash: float, spacing: float, size: tuple[int, int], ground: tuple[int, int],
           sun=(0.3, 0.1)) -> np.ndarray:
    n, sh, sw = vol.shape[:3]
    cw, ch = size
    gx, gy = ground
    cx, cy = (sw - 1) * 0.5, (sh - 1) * 0.5
    c, s = math.cos(turn), math.sin(turn)
    # destination grid in art pixels, relative to the ground point
    xs = np.arange(cw)[None, :] - gx
    ys = np.arange(ch)[:, None] - gy

    def sample(z: int, lift: float, offx: float = 0.0):
        dy = (ys + lift) / squash
        dx = xs - offx
        # inverse rotation: destination (dx, dy) -> source (u, v)
        u = c * dx + s * dy
        v = -s * dx + c * dy
        # floor(x + 0.5) with a tiny bias, never np.rint: round-half-to-even on an even-sized
        # slice (centre at k + 0.5) maps two screen columns onto one source column and drops the next
        ui = np.floor(u + cx + 0.5 + 1e-4).astype(int)
        vi = np.floor(v + cy + 0.5 + 1e-4).astype(int)
        ok = (ui >= 0) & (ui < sw) & (vi >= 0) & (vi < sh)
        out = np.zeros((ch, cw, 4), np.uint8)
        out[ok] = vol[z, vi[ok], ui[ok]]
        return out

    frame = np.zeros((ch, cw, 4), np.uint8)
    frame[..., :3] = BG
    frame[..., 3] = 255

    # ground shadow: each slice's footprint pushed along the sun vector on the ground plane
    shadow = np.zeros((ch, cw), bool)
    for z in range(0, n, 1):
        m = sample(z, 0.0, offx=z * sun[0])[..., 3] > 0
        sy = int(round(z * sun[1]))
        if sy:  # shift down without wrap-around
            m = np.concatenate([np.zeros((sy, cw), bool), m[:-sy]], axis=0)
        shadow |= m
    # soft edge: blur the single union mask, then blend toward BG in 3 flat steps (no dither noise)
    soft = shadow.astype(float)
    for _ in range(2):
        q = np.pad(soft, 1, mode="edge")
        soft = sum(q[dy : dy + ch, dx : dx + cw] for dy in range(3) for dx in range(3)) / 9
    level = np.select([soft > 0.7, soft > 0.4, soft > 0.12], [1.0, 0.6, 0.3], 0.0)
    mix = np.array(BG) + (np.array(SHADOW) - np.array(BG)) * level[..., None]
    frame[..., :3] = np.rint(mix).astype(np.uint8)

    # stack, bottom to top, into a separate buffer so the outline traces only the stack
    stack = np.zeros((ch, cw, 4), np.uint8)
    for z in range(n):
        sl = sample(z, z * spacing)
        m = sl[..., 3] > 0
        stack[m] = sl[m]
    alpha = stack[..., 3] > 0
    p = np.pad(alpha, 1)
    ring = ~alpha & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])
    frame[alpha, :3] = stack[alpha, :3]
    frame[ring, :3] = INK
    return frame[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", type=Path, default=HERE / "slices.png")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--squash", type=float, default=0.6)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--scale", type=int, default=0, help="0: 4x, or 3x if 4x is wider than 900 px")
    ap.add_argument("--offset", type=float, default=45.0, help="turn at frame 0, degrees (45: entrance corner faces the viewer)")
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--out", type=Path, default=HERE / "turntable.gif")
    ap.add_argument("--check", type=Path, default=HERE / "frames_check.png")
    args = ap.parse_args()

    vol = load_strip(args.strip, args.count)
    n, sh, sw = vol.shape[:3]
    diag = math.ceil(math.hypot(sw, sh))
    pad = 6
    shadow_len = int(math.ceil(n * 0.3)) + 3
    cw = diag + shadow_len + 2 * pad
    ch = int(math.ceil(diag * args.squash)) + n + 2 * pad
    # ground point fixed for every frame: no wobble at the base
    ground = (pad + diag // 2, ch - pad - int(math.ceil(diag * args.squash / 2)))

    if args.scale == 0:
        args.scale = 4 if cw * 4 <= 900 else 3
    frames = []
    for i in range(args.frames):
        turn = 2 * math.pi * i / args.frames + math.radians(args.offset)
        f = render(vol, turn, args.squash, args.spacing, (cw, ch), ground)
        frames.append(Image.fromarray(f).resize((cw * args.scale, ch * args.scale), Image.NEAREST))

    # one shared palette across all frames so colours do not flicker
    pal_src = Image.new("RGB", (cw * args.scale, ch * args.scale * len(frames)))
    for i, f in enumerate(frames):
        pal_src.paste(f, (0, i * ch * args.scale))
    pal = pal_src.quantize(colors=64, method=Image.MEDIANCUT, dither=Image.NONE)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=int(round(1000 / args.fps)), loop=0,
              disposal=1, optimize=False)

    # not symmetric: check six angles in a 3x2 grid
    angles = (0, 45, 90, 135, 180, 270)
    picks = [round(a / 360 * args.frames) for a in angles]
    small = [frames[i] for i in picks]
    gap = 8
    fw, fh = small[0].size
    sheet = Image.new("RGB", (3 * fw + 2 * gap, 2 * fh + gap), (255, 255, 255))
    for k, im in enumerate(small):
        sheet.paste(im, ((k % 3) * (fw + gap), (k // 3) * (fh + gap)))
    sheet.save(args.check)
    print(f"{args.out} {len(frames)} frames {frames[0].size} | check {args.check} at frames {picks}")


if __name__ == "__main__":
    main()
