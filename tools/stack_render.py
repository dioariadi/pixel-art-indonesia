"""Sprite-stack turntable renderer, version 2 (shared by every model folder).

Input: a material volume `model.npz` (array `mat`, uint8 [z, y, x], 0 = empty) and `materials.json`
next to it: {"<id>": {"name": str, "ramp": [[r,g,b] x 4, dark -> light], "emissive": bool}}.
Legacy input: `--strip slices.png --count N` (colour-only slices; each colour becomes a flat material).

Per frame, all at 1x art pixels:
  * stack the slices bottom to top (rotate by the turn, squash y for pitch, lift `spacing` px per slice) into an
    index buffer. Each slice is sampled at `substeps` fractional heights (lowest first) so it acts as a slab
    and thin parts do not break into stripes at low pitch; --substeps 1 is the old one-sample behaviour: for each screen pixel the voxel drawn there. Painting by z is exact depth order for this
    projection (a higher voxel landing on the same pixel is always nearer the camera).
  * from that buffer: material, height (slice), depth toward the camera, surface normal, occlusion.
  * light: normals rotated by the turn, lit by a lamp fixed in screen space, quantised to 4 ramp steps;
    occluded voxels drop one step. Emissive and legacy-flat materials ignore both.
  * lines: outer silhouette (ink), depth jumps (ink, on the farther pixel), material changes without a depth
    jump (darkest ramp colour of the darker material, on its side).
Axes: x right, y toward the viewer at turn 0, z up.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (214, 230, 242)
SHADOW = (176, 194, 212)
INK = (24, 28, 48)
SHADOW_LEVELS = (0.3, 0.6, 1.0)


def shadow_colours():
    """The flat shadow blends drawn on the ground (used by check_model.py to tell ground from model)."""
    bg, sh = np.array(BG, float), np.array(SHADOW, float)
    return [tuple(int(v) for v in np.rint(bg + (sh - bg) * lv)) for lv in SHADOW_LEVELS]


# ----------------------------------------------------------------------------------------------- input
@dataclass
class Model:
    mat: np.ndarray                       # uint8 [z, y, x]
    ramps: np.ndarray                     # uint8 [256, 4, 3]
    emissive: np.ndarray                  # bool [256]: fixed tone, no light, no occlusion
    flat: np.ndarray                      # bool [256]: legacy colour, no light, no occlusion
    names: dict = field(default_factory=dict)
    legacy: bool = False


def load_model(npz: Path, materials: Path | None = None) -> Model:
    mat = np.load(npz)["mat"].astype(np.uint8)
    materials = materials or npz.with_name("materials.json")
    spec = json.loads(Path(materials).read_text())
    ramps = np.zeros((256, 4, 3), np.uint8)
    emissive = np.zeros(256, bool)
    names = {}
    for k, v in spec.items():
        i = int(k)
        r = np.array(v["ramp"], np.uint8)
        assert r.shape == (4, 3), f"material {k}: ramp must be 4 RGB colours, got {r.shape}"
        ramps[i] = r
        emissive[i] = bool(v.get("emissive", False))
        names[i] = v.get("name", str(i))
    missing = sorted(set(np.unique(mat)) - {0} - set(names))
    assert not missing, f"materials.json has no entry for ids {missing}"
    return Model(mat, ramps, emissive, np.zeros(256, bool), names)


def load_strip(path: Path, count: int) -> Model:
    a = np.array(Image.open(path).convert("RGBA"))
    h, total_w = a.shape[:2]
    assert total_w % count == 0, f"strip width {total_w} is not a multiple of {count}"
    w = total_w // count
    vol = np.stack([a[:, i * w:(i + 1) * w] for i in range(count)])          # [z, y, x, 4]
    filled = vol[..., 3] > 0
    cols, inv = np.unique(vol[filled][:, :3], axis=0, return_inverse=True)
    assert len(cols) <= 255, f"legacy strip has {len(cols)} colours, max 255"
    mat = np.zeros(filled.shape, np.uint8)
    mat[filled] = inv.ravel() + 1
    ramps = np.zeros((256, 4, 3), np.uint8)
    ramps[1:len(cols) + 1] = cols[:, None, :]
    flat = np.zeros(256, bool)
    flat[1:len(cols) + 1] = True
    return Model(mat, ramps, np.zeros(256, bool), flat, {i + 1: f"c{i + 1}" for i in range(len(cols))}, legacy=True)


# ------------------------------------------------------------------------------------- precompute
def box_blur3(a: np.ndarray, times: int) -> np.ndarray:
    """Separable 3-tap box blur along all three axes, zero outside the grid."""
    for _ in range(times):
        for ax in range(3):
            p = np.pad(a, [(1, 1) if i == ax else (0, 0) for i in range(3)])
            sl = [slice(None)] * 3
            parts = []
            for o in range(3):
                sl[ax] = slice(o, o + a.shape[ax])
                parts.append(p[tuple(sl)])
            a = (parts[0] + parts[1] + parts[2]) / 3.0
    return a


def surface_normals(occ: np.ndarray, blur: int = 2) -> np.ndarray:
    """Unit outward normals [z, y, x, 3] as (nx, ny, nz) from the gradient of the blurred occupancy."""
    b = box_blur3(occ.astype(np.float32), blur)
    gz, gy, gx = np.gradient(b)
    n = -np.stack([gx, gy, gz], -1).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return n


def surface_mask(occ: np.ndarray) -> np.ndarray:
    p = np.pad(occ, 1)
    inner = (occ & p[:-2, 1:-1, 1:-1] & p[2:, 1:-1, 1:-1] & p[1:-1, :-2, 1:-1] & p[1:-1, 2:, 1:-1]
             & p[1:-1, 1:-1, :-2] & p[1:-1, 1:-1, 2:])
    return occ & ~inner


def occlusion(occ: np.ndarray, n: np.ndarray, radius: int = 3) -> np.ndarray:
    """Contact shadow per surface voxel: how filled the space just outside it is. Reads a blurred occupancy
    (box radius ~`radius`/2+1) at the point `radius` voxels out along the normal, then averages that value over
    neighbouring surface voxels so creases come out as smooth bands, not specks. 0..1, 0 = open air.
    Independent of the turn angle."""
    surf = surface_mask(occ)
    zi, yi, xi = np.nonzero(surf)
    nn = n[zi, yi, xi]
    blur = box_blur3(occ.astype(np.float32), max(1, radius // 2 + 1))
    sz, sy, sx = occ.shape
    pz = np.clip(np.floor(zi + radius * nn[:, 2] + 0.5).astype(int), 0, sz - 1)
    py = np.clip(np.floor(yi + radius * nn[:, 1] + 0.5).astype(int), 0, sy - 1)
    px = np.clip(np.floor(xi + radius * nn[:, 0] + 0.5).astype(int), 0, sx - 1)
    raw = np.zeros(occ.shape, np.float32)
    raw[zi, yi, xi] = blur[pz, py, px]
    # average over surface neighbours (3x3x3), surface voxels only
    num = box_blur3(raw, 1)
    den = box_blur3(surf.astype(np.float32), 1)
    out = np.where(surf, num / np.maximum(den, 1e-6), 0).astype(np.float32)
    return out


def drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove 8-connected pieces of a line mask smaller than min_px pixels."""
    if min_px <= 1 or not mask.any():
        return mask
    h, w = mask.shape
    out = mask.copy()
    seen = np.zeros_like(mask)
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack, comp = [(y0, x0)], []
        seen[y0, x0] = True
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        if len(comp) < min_px:
            for y, x in comp:
                out[y, x] = False
    return out


@dataclass
class Config:
    frames: int = 48
    squash: float = 0.6
    spacing: float = 1.0
    substeps: int = 0                     # samples per slice; 0 = auto_substeps(squash, spacing), 1 = v1 behaviour
    scale: int = 4
    fps: float = 12
    # lamp in camera space: (right, up on screen, toward viewer)
    light: tuple = (-0.5, 0.5, 0.7)
    tone_thresholds: tuple = (-0.3, 0.1, 0.55)      # lit = n.L, cut into ramp steps 0..3
    normal_blur: int = 2
    ao: bool = True
    ao_radius: int = 3
    ao_thresh: float = 0.08
    despeckle: bool = True
    silhouette: bool = True
    depth_lines: bool = True
    depth_thresh: float = 3.0
    slope_k: float = 1.0
    min_line: int = 4
    material_lines: bool = True
    sun: tuple = (0.3, 0.1)


class Prepared:
    """Everything that does not depend on the turn angle."""

    def __init__(self, model: Model, cfg: Config):
        self.model = model
        self.cfg = cfg
        mat = model.mat
        self.n, self.sh, self.sw = mat.shape
        occ = mat > 0
        self.occ = occ
        self.mat_flat = mat.ravel()
        self.occ_flat = occ.ravel()
        nrm = surface_normals(occ, cfg.normal_blur)
        self.nrm_flat = nrm.reshape(-1, 3)
        fixed = model.emissive | model.flat
        self.fixed = fixed
        if cfg.ao and not model.legacy:
            ao = occlusion(occ, nrm, cfg.ao_radius)
            self.ao_flat = (ao > cfg.ao_thresh).ravel()
            self.ao_raw = ao
        else:
            self.ao_flat = np.zeros(occ.size, bool)
            self.ao_raw = np.zeros(occ.shape, np.float32)
        # material luminance (mean of the ramp) decides which side of a material line gets the line
        self.lum = (model.ramps.astype(float) @ [0.299, 0.587, 0.114]).mean(1)
        # layout: same as v1 render_gif.py so frames line up with old GIFs
        diag = math.ceil(math.hypot(self.sw, self.sh))
        pad = 6
        shadow_len = int(math.ceil(self.n * cfg.sun[0])) + 3
        self.maxlift = int(math.floor((self.n - 1) * cfg.spacing + 0.5))
        self.cw = diag + shadow_len + 2 * pad
        self.ch = int(math.ceil(diag * cfg.squash)) + self.maxlift + 1 + 2 * pad
        self.ground = (pad + diag // 2, self.ch - pad - int(math.ceil(diag * cfg.squash / 2)))
        self.lifts = [int(math.floor(z * cfg.spacing + 0.5)) for z in range(self.n)]
        self.nonempty = [z for z in range(self.n) if occ[z].any()]
        self.substeps = cfg.substeps if cfg.substeps > 0 else auto_substeps(cfg.squash, cfg.spacing)
        # light in model-upright screen frame (x right, y' toward viewer, z up)
        q, sp = cfg.squash, cfg.spacing
        hn = math.hypot(q, sp)
        up = np.array([0.0, -q, sp]) / hn
        toward = np.array([0.0, sp, q]) / hn
        L = cfg.light[0] * np.array([1.0, 0, 0]) + cfg.light[1] * up + cfg.light[2] * toward
        self.L = L / np.linalg.norm(L)
        self.toward = toward
        self.up = up


# ------------------------------------------------------------------------------------------ frame
def _source_offsets(u, v, cx, cy, sw, sh):
    """Flat in-slice voxel offset (y * sw + x) read by each screen pixel, -1 outside the grid.

    u, v are measured from the grid centre. A sample that lands on a voxel boundary (within 1e-4) goes to the
    voxel nearer the centre. v1 always rounded such ties up (floor(t + 1e-4)), which tips opposite ways at 0 and
    180 degrees, so a model could gain a row at one and lose it at the other; this rule is mirror-symmetric."""
    ui = _toward_centre(u, cx)
    vi = _toward_centre(v, cy)
    ok = (ui >= 0) & (ui < sw) & (vi >= 0) & (vi < sh)
    return np.where(ok, vi * sw + ui, -1)


def _toward_centre(w, c):
    t = w + c + 0.5                                   # voxel coordinate: voxel i spans [i, i + 1)
    return np.where(w < 0, np.floor(t + 1e-4), np.ceil(t - 1e-4) - 1).astype(np.int64)


def auto_substeps(squash: float, spacing: float) -> int:
    """Samples per slice so that consecutive samples are at most one voxel's screen height apart."""
    return max(2, math.ceil(spacing / max(0.25, squash)))


def render_frame(P: Prepared, turn: float) -> dict:
    cfg = P.cfg
    cw, ch = P.cw, P.ch
    gx, gy = P.ground
    sw, sh = P.sw, P.sh
    S = sw * sh
    cx, cy = (sw - 1) * 0.5, (sh - 1) * 0.5
    c, s = math.cos(turn), math.sin(turn)

    # source pixel for every screen pixel of an extended canvas; slice z reads rows [lift, lift + ch)
    xs = np.arange(cw)[None, :] - gx
    ys = (np.arange(ch + P.maxlift + 1)[:, None] - gy) / cfg.squash
    u = c * xs + s * ys
    v = -s * xs + c * ys
    ofs = _source_offsets(u, v, cx, cy, sw, sh)

    # each slice is a slab: sample it at `substeps` fractional heights z + k/n (lift + k/n * spacing),
    # lowest first, so thin parts at low pitch do not leave background rows between slices.
    # One offset map per fractional height; the integer part of the lift stays a row shift.
    nsub = P.substeps
    sub_ofs = [ofs]
    for k in range(1, nsub):
        ys_k = (np.arange(ch + P.maxlift + 1)[:, None] - gy + k / nsub * cfg.spacing) / cfg.squash
        sub_ofs.append(_source_offsets(c * xs + s * ys_k, -s * xs + c * ys_k, cx, cy, sw, sh))

    idx = np.full((ch, cw), -1, np.int64)
    for z in P.nonempty:
        lf = P.lifts[z]
        for so in sub_ofs:
            f = so[lf:lf + ch]
            g = np.where(f >= 0, z * S + f, 0)
            hit = (f >= 0) & P.occ_flat[g]
            idx[hit] = g[hit]

    # ground shadow: each slice footprint pushed along the sun on the ground plane
    shadow = np.zeros((ch, cw), bool)
    base = ofs[:ch]
    for z in P.nonempty:
        fp = (base >= 0) & P.occ_flat[np.where(base >= 0, z * S + base, 0)]
        sx, sy = int(math.floor(z * cfg.sun[0] + 0.5)), int(math.floor(z * cfg.sun[1] + 0.5))
        if sx or sy:
            m = np.zeros_like(fp)
            m[sy:, sx:] = fp[:ch - sy, :cw - sx]
            fp = m
        shadow |= fp
    soft = shadow.astype(float)
    for _ in range(2):
        q = np.pad(soft, 1, mode="edge")
        soft = sum(q[a:a + ch, b:b + cw] for a in range(3) for b in range(3)) / 9
    level = np.select([soft > 0.7, soft > 0.4, soft > 0.12], [1.0, 0.6, 0.3], 0.0)
    bgimg = np.rint(np.array(BG) + (np.array(SHADOW) - np.array(BG)) * level[..., None]).astype(np.uint8)

    # buffers
    filled = idx >= 0
    gi = np.where(filled, idx, 0)
    zb = gi // S
    vb = (gi % S) // sw
    ub = gi % sw
    mat = np.where(filled, P.mat_flat[gi], 0).astype(np.int64)
    height = np.where(filled, zb, -1)
    vp = s * (ub - cx) + c * (vb - cy)                                    # toward the viewer, in voxels
    depth = np.where(filled, (cfg.spacing * vp + cfg.squash * zb) / math.hypot(cfg.spacing, cfg.squash), -np.inf)

    # lighting with normals rotated into the screen frame
    nm = P.nrm_flat[gi]
    nx = c * nm[..., 0] - s * nm[..., 1]
    ny = s * nm[..., 0] + c * nm[..., 1]
    lit = nx * P.L[0] + ny * P.L[1] + nm[..., 2] * P.L[2]
    tone_light = np.digitize(lit, cfg.tone_thresholds)                  # 0..3
    fixed = P.fixed[mat]
    tone_light = np.where(fixed, 2, tone_light)
    if cfg.despeckle:
        tone_light = despeckle(tone_light, mat, filled)
    aodrop = filled & P.ao_flat[gi] & ~fixed
    tone = np.clip(tone_light - aodrop, 0, 3)
    tone = np.where(fixed, 2, tone)

    # lines
    # expected depth change per screen pixel on a smooth surface, from the normal in camera space
    nt = nx * 0 + ny * P.toward[1] + nm[..., 2] * P.toward[2]
    nu = ny * P.up[1] + nm[..., 2] * P.up[2]
    ntc = np.maximum(nt, 0.2)
    slope_x = np.abs(nx) / ntc
    slope_y = np.abs(nu) / ntc
    ring, dline, mline = outline_masks(P, filled, depth, mat, slope_x, slope_y)

    col = P.model.ramps[mat, tone]
    frame = bgimg.copy()
    frame[filled] = col[filled]
    if cfg.material_lines:
        frame[mline] = P.model.ramps[mat, 0][mline]
    if cfg.depth_lines:
        frame[dline] = INK
    if cfg.silhouette:
        frame[ring] = INK
    return dict(frame=frame, idx=idx, filled=filled, mat=mat, height=height, depth=depth, tone_light=tone_light,
                aodrop=aodrop, ring=ring, dline=dline, mline=mline, bg=bgimg)


def despeckle(tone, mat, filled):
    """A pixel whose 4 same-material neighbours (at least 3 present) all agree on another tone takes it."""
    t = np.pad(tone, 1, constant_values=-1)
    m = np.pad(np.where(filled, mat, -1), 1, constant_values=-1)
    nbt = np.stack([t[:-2, 1:-1], t[2:, 1:-1], t[1:-1, :-2], t[1:-1, 2:]])
    nbm = np.stack([m[:-2, 1:-1], m[2:, 1:-1], m[1:-1, :-2], m[1:-1, 2:]])
    same = nbm == mat[None]
    out = tone.copy()
    for cand in range(4):
        agree = (same & (nbt == cand)).sum(0)
        n_same = same.sum(0)
        flip = filled & (n_same >= 3) & (agree == n_same) & (tone != cand)
        out[flip] = cand
    return out


def outline_masks(P: Prepared, filled, depth, mat, slope_x, slope_y):
    """Depth line on the farther pixel when the nearer neighbour sits more than depth_thresh voxels in front
    of what the farther pixel's own surface slope predicts (so grazing curves do not draw lines)."""
    cfg = P.cfg
    fp = np.pad(filled, 1)
    ring = ~filled & (fp[:-2, 1:-1] | fp[2:, 1:-1] | fp[1:-1, :-2] | fp[1:-1, 2:])
    dp = np.pad(depth, 1, constant_values=-np.inf)
    mp = np.pad(mat, 1)
    dline = np.zeros_like(filled)
    mline = np.zeros_like(filled)
    lum = P.lum
    emis = P.model.emissive
    for nb_d, nb_m, sl in ((dp[:-2, 1:-1], mp[:-2, 1:-1], slope_y), (dp[2:, 1:-1], mp[2:, 1:-1], slope_y),
                           (dp[1:-1, :-2], mp[1:-1, :-2], slope_x), (dp[1:-1, 2:], mp[1:-1, 2:], slope_x)):
        both = filled & (nb_m > 0)
        with np.errstate(invalid="ignore"):
            diff = np.where(both, nb_d - depth, 0.0)   # > 0: neighbour is nearer the camera, this pixel farther
        dline |= both & (diff > cfg.depth_thresh + cfg.slope_k * sl)
        darker = (lum[mat] < lum[nb_m]) | ((lum[mat] == lum[nb_m]) & (mat < nb_m))
        mline |= both & (np.abs(diff) <= cfg.depth_thresh) & (nb_m != mat) & darker & ~emis[mat]
    if P.model.legacy:
        mline[:] = False
    dline = drop_small(dline, cfg.min_line)
    mline &= ~dline
    mline = drop_small(mline, cfg.min_line)
    return ring, dline, mline


# ------------------------------------------------------------------------------------------ output
def save_gif(frames: list[np.ndarray], path: Path, scale: int, fps: float):
    imgs = [Image.fromarray(f) for f in frames]
    allc = np.unique(np.concatenate([f.reshape(-1, 3) for f in frames]), axis=0)
    h, w = frames[0].shape[:2]
    if len(allc) <= 256:
        pal_img = Image.new("P", (1, 1))
        flat = list(allc.ravel()) + [0] * (768 - allc.size)
        pal_img.putpalette([int(x) for x in flat])
        q = [im.quantize(palette=pal_img, dither=Image.Dither.NONE) for im in imgs]
    else:
        src = Image.new("RGB", (w, h * len(imgs)))
        for i, im in enumerate(imgs):
            src.paste(im, (0, i * h))
        pal = src.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q = [im.resize((w * scale, h * scale), Image.NEAREST) for im in q]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(round(1000 / fps)), loop=0, disposal=1,
              optimize=False)
    return len(allc)


def save_check(frames, path: Path, nframes: int):
    picks = [k * nframes // 8 for k in range(8)]
    h, w = frames[0].shape[:2]
    small = [Image.fromarray(frames[i]).resize((w * 3, h * 3), Image.NEAREST) for i in picks]
    gap = 8
    fw, fh = small[0].size
    sheet = Image.new("RGB", (4 * fw + 3 * gap, 2 * fh + gap), (255, 255, 255))
    for k, im in enumerate(small):
        sheet.paste(im, ((k % 4) * (fw + gap), (k // 4) * (fh + gap)))
    sheet.save(path)
    return picks


def _font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    except OSError:
        return ImageFont.load_default()


def save_debug(P: Prepared, angle_deg: float, path: Path, scale: int = 6):
    r = render_frame(P, math.radians(angle_deg))
    filled, mat = r["filled"], r["mat"]
    ramps = P.model.ramps
    white = np.full(r["frame"].shape, 255, np.uint8)
    flat = white.copy()
    flat[filled] = ramps[mat, 2][filled]
    greys = np.array([[60] * 3, [120] * 3, [180] * 3, [235] * 3], np.uint8)
    light = white.copy()
    light[filled] = greys[r["tone_light"]][filled]
    light[filled & P.fixed[mat]] = (250, 120, 160)
    occl = white.copy()
    occl[filled] = (225, 225, 225)
    occl[r["aodrop"]] = (90, 90, 110)
    lines = white.copy()
    lines[filled] = (235, 235, 235)
    lines[r["mline"]] = ramps[mat, 0][r["mline"]]
    lines[r["dline"]] = INK
    lines[r["ring"]] = (150, 150, 160)
    panels = [("flat materials", flat), ("lighting only", light), ("occlusion only", occl),
              ("outlines (grey=silhouette)", lines), ("final", r["frame"])]
    h, w = flat.shape[:2]
    lab, gap = 30, 10
    sheet = Image.new("RGB", (len(panels) * (w * scale + gap) + gap, h * scale + lab + gap), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    f = _font(15)
    for k, (name, img) in enumerate(panels):
        x0 = gap + k * (w * scale + gap)
        sheet.paste(Image.fromarray(img).resize((w * scale, h * scale), Image.NEAREST), (x0, lab))
        d.text((x0, 6), f"{name} @ {angle_deg:g} deg", fill=(20, 20, 30), font=f)
    sheet.save(path)


def render_all(P: Prepared):
    return [render_frame(P, 2 * math.pi * i / P.cfg.frames)["frame"] for i in range(P.cfg.frames)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", nargs="?", type=Path, help="model.npz (materials.json beside it)")
    ap.add_argument("--materials", type=Path)
    ap.add_argument("--strip", type=Path, help="legacy colour-only slice strip")
    ap.add_argument("--count", type=int, help="slice count of the legacy strip")
    ap.add_argument("--out", type=Path, help="GIF path (default turntable.gif beside the input)")
    ap.add_argument("--check", type=Path, help="8-angle sheet (default frames_check.png beside the GIF)")
    ap.add_argument("--debug", nargs="?", const=True, default=None,
                    help="write a debug sheet (optional path; default debug_sheet.png beside the GIF)")
    ap.add_argument("--debug-angle", type=float, default=45.0)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--squash", type=float, default=0.6)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--pitch", type=float, help="camera pitch in degrees: squash = sin, spacing = cos "
                    "(overrides --squash and --spacing)")
    ap.add_argument("--substeps", default="auto",
                    help="samples per slice: 'auto' = max(2, ceil(spacing / max(0.25, squash))), an integer forces "
                         "it, 1 = old single-sample output")
    ap.add_argument("--light", type=float, nargs=3, default=(-0.5, 0.5, 0.7),
                    help="lamp in screen space: right, up, toward viewer")
    ap.add_argument("--tones", type=float, nargs=3, default=(-0.3, 0.1, 0.55), help="n.L cuts between ramp steps")
    ap.add_argument("--normal-blur", type=int, default=2)
    ap.add_argument("--depth-thresh", type=float, default=3.0)
    ap.add_argument("--slope-k", type=float, default=1.0,
                    help="extra depth allowance per unit of the farther surface's own slope")
    ap.add_argument("--min-line", type=int, default=4, help="drop inner line pieces shorter than this (px)")
    ap.add_argument("--ao-thresh", type=float, default=0.08)
    ap.add_argument("--ao-radius", type=int, default=3)
    ap.add_argument("--no-ao", action="store_true")
    ap.add_argument("--no-despeckle", action="store_true")
    ap.add_argument("--no-silhouette", action="store_true")
    ap.add_argument("--no-depth-lines", action="store_true")
    ap.add_argument("--no-material-lines", action="store_true")
    a = ap.parse_args(argv)

    if a.strip:
        if not a.count:
            ap.error("--strip needs --count")
        model = load_strip(a.strip, a.count)
        src = a.strip
    elif a.model:
        model = load_model(a.model, a.materials)
        src = a.model
    else:
        ap.error("give model.npz or --strip/--count")
    if a.pitch is not None:
        assert 0 < a.pitch < 90, "--pitch must be between 0 and 90 degrees"
        a.squash, a.spacing = math.sin(math.radians(a.pitch)), math.cos(math.radians(a.pitch))
    if a.substeps == "auto":
        substeps = 0
    else:
        try:
            substeps = int(a.substeps)
        except ValueError:
            ap.error("--substeps must be 'auto' or a positive integer")
        if substeps < 1:
            ap.error("--substeps must be 'auto' or a positive integer")
    out = a.out or src.with_name("turntable.gif")
    check = a.check or out.with_name("frames_check.png")
    cfg = Config(frames=a.frames, squash=a.squash, spacing=a.spacing, substeps=substeps, scale=a.scale, fps=a.fps,
                 light=tuple(a.light), tone_thresholds=tuple(sorted(a.tones)), normal_blur=a.normal_blur,
                 ao=not a.no_ao, ao_radius=a.ao_radius, ao_thresh=a.ao_thresh, despeckle=not a.no_despeckle,
                 silhouette=not a.no_silhouette, depth_lines=not a.no_depth_lines, depth_thresh=a.depth_thresh,
                 slope_k=a.slope_k, min_line=a.min_line,
                 material_lines=not a.no_material_lines)
    t0 = time.time()
    P = Prepared(model, cfg)
    t1 = time.time()
    frames = render_all(P)
    t2 = time.time()
    ncol = save_gif(frames, out, cfg.scale, cfg.fps)
    picks = save_check(frames, check, cfg.frames)
    msg = (f"{out} {len(frames)} frames {frames[0].shape[1]}x{frames[0].shape[0]} art px x{cfg.scale}, "
           f"{ncol} colours | squash {cfg.squash:.3f} spacing {cfg.spacing:.3f} substeps {P.substeps} | check {check} at frames {picks} | prep {t1 - t0:.1f}s render {t2 - t1:.1f}s")
    if a.debug:
        dpath = Path(a.debug) if isinstance(a.debug, str) else out.with_name(out.stem + "_debug.png")
        save_debug(P, a.debug_angle, dpath)
        msg += f" | debug {dpath}"
    print(msg)


if __name__ == "__main__":
    main()
