"""Candi Prambanan central court (v2) as a material volume, written as a sprite-stack slice strip
(ground slice on the left). Built directly in NumPy, no 4-view carve.

v2 follows the user references in refs/: user_aerial_layout.png (lower half, 2026) for the layout,
user_group_photo.png for silhouettes, the two illustrations for how simplification stays readable.
Temples are stout (width/height about 0.7): wide stepped plinth, cross-plan body with a porch on each
side, broad stepped tower of short tiers edged with finials, one big finial on the peak.

Axes: x right, y toward the viewer at turn 0, z up. Grid is even-sized, so the turn axis sits on a pixel
corner; temple centres are whole integers, so half-width h gives a 2h px footprint.
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
W = 160
NZ = 70
CX = CY = (W - 1) / 2
BASE = 3            # temples stand on the paving (court platform z 1..2)

PAL = {
    1: (104, 138, 84),    # ground green
    2: (204, 196, 180),   # court paving, pale warm grey
    3: (164, 148, 126),   # stone: upward faces, finial tips, door frames (light warm)
    4: (122, 109, 96),   # stone: walls facing the light (+y, -x)
    5: (92, 83, 76),     # stone: walls facing away (-y, +x)
    6: (64, 58, 54),      # stone: recesses under cornices
    7: (30, 26, 26),      # doorways
}

ZZ, YY, XX = np.mgrid[0:NZ, 0:W, 0:W]
DX = (XX - CX).astype(float)
DY = (YY - CY).astype(float)

S = np.zeros((NZ, W, W), bool)
MAT = np.zeros((NZ, W, W), np.uint8)   # 1 ground, 2 paving, 3 stone, 4 finial tip, 5 door, 6 door frame


def put(mask, mat):
    S[mask] = True
    MAT[mask] = mat


def clear(mask):
    S[mask] = False
    MAT[mask] = 0


# ---------------- ground plate, court, wall with four gates ----------------
put(ZZ == 0, 1)
CH = 74                                     # court half-size
AXG, AYG = np.abs(DX), np.abs(DY)
put((ZZ >= 1) & (ZZ <= 2) & (AXG < CH + 2) & (AYG < CH + 2), 2)
wall = (np.maximum(AXG, AYG) >= CH) & (np.maximum(AXG, AYG) < CH + 2) & (ZZ >= 3) & (ZZ <= 5)
gate = np.minimum(AXG, AYG) < 7
put(wall & ~gate, 3)
for along, across in ((AXG, AYG), (AYG, AXG)):     # two steps down outside each gate
    put((across < 7) & (along >= CH + 2) & (along < CH + 3) & (ZZ >= 1) & (ZZ <= 2), 2)
    put((across < 7) & (along >= CH + 3) & (along < CH + 4) & (ZZ == 1), 2)


def finial(fx, fy, z0, big):
    """Small pointed finial: 2x2, 2-3 layers. Big (corner): 4x4 foot, 2x2 shaft, light tip."""
    m2 = (np.abs(DX - fx) < 1) & (np.abs(DY - fy) < 1)
    if big:
        m4 = (np.abs(DX - fx) < 2) & (np.abs(DY - fy) < 2)
        put(m4 & (ZZ == z0), 3)
        put(m2 & (ZZ >= z0 + 1) & (ZZ <= z0 + 2), 3)
        put(m2 & (ZZ == z0 + 3), 4)
    else:
        put(m2 & (ZZ == z0), 3)
        put(m2 & (ZZ == z0 + 1), 4)


def edge_positions(d, spacing=4):
    n = max(1, round(2 * d / spacing))
    return np.unique(np.round(np.linspace(-d, d, n + 1)).astype(int))


def temple(cx, cy, H, steps, p, core, pj, pw, bh, tiers, crown, stairs, ext, sw):
    """steps: plinth half-widths bottom-up (p layers in all); core: body half-width; pj/pw: porch
    projection and half-width; bh: body height; tiers: tower tier half-widths; crown: peak finial
    half-width; stairs: (sx, sy) sides with a door and stairway; ext: stair run beyond the plinth."""
    dx, dy = DX - cx, DY - cy
    ax, ay = np.abs(dx), np.abs(dy)
    lz = ZZ - BASE
    sq = lambda h: (ax < h) & (ay < h)
    cross = lambda h, j, w: sq(h) | ((ax < h + j) & (ay < w)) | ((ay < h + j) & (ax < w))

    cuts = np.linspace(0, p, len(steps) + 1).round().astype(int)
    for h, z0, z1 in zip(steps, cuts[:-1], cuts[1:]):
        put(cross(h, 1, h // 2) & (lz >= z0) & (lz < z1), 3)
    ptop = steps[-1]

    # body: cross plan, porch on every side, cornice band on top
    put(cross(core, pj, pw) & (lz >= p) & (lz < p + bh), 3)
    put(cross(core + 1, pj, pw + 1) & (lz == p + bh), 3)

    for sx, sy in stairs:
        along = dx * sx + dy * sy
        across = np.abs(dx * sy - dy * sx)
        face = core + pj                       # porch outer face is at along < face
        dw = max(1, pw - 2)
        dh = max(3, round(bh * 0.6))
        # lighter frame one px proud of the porch face, then a dark opening recessed 1 px
        put((along >= face) & (along < face + 1) & (across < dw + 1) & (lz >= p) & (lz < p + dh + 1), 6)
        opening = (across < dw) & (lz >= p) & (lz < p + dh)
        clear(opening & (along >= face - 1) & (along < face + 1))
        put(opening & (along >= face - 2) & (along < face - 1), 5)
        # stairway from the paving up to the plinth top
        st = (across < sw) & (along >= 0) & (along < steps[0] + ext)
        hz = np.where(along < steps[0], p, np.ceil(p * (steps[0] + ext - along) / (ext + 1)))
        put(st & (lz >= 0) & (lz < hz) & ~((along >= face - 2) & (along < face + 1) & (lz >= p)), 3)

    # ring of finials on the plinth gallery, skipped in front of the porches
    d = ptop - 1
    if d - 1 >= core:
        for a in edge_positions(d):
            if abs(a) <= pw + 1:
                continue
            for fx, fy in ((a, d), (a, -d), (d, a), (-d, a)):
                finial(cx + fx, cy + fy, BASE + p, False)

    # tower: short tiers, each a cross-plan block + overhanging cornice, small finials along the edge
    ch = crown_h = max(4, round(0.15 * H))
    z = p + bh + 1
    tz = np.linspace(z, H - ch, len(tiers) + 1).round().astype(int)
    for h, z0, z1 in zip(tiers, tz[:-1], tz[1:]):
        w = max(1, round(h * 0.4))
        put(cross(h, 1, w) & (lz >= z0) & (lz < z1 - 1), 3)
        put(sq(h + 1) & (lz == z1 - 1), 3)
        for a in edge_positions(h, 4 if h >= 6 else 3):
            if abs(a) == h:
                continue
            for fx, fy in ((a, h), (a, -h), (h, a), (-h, a)):
                finial(cx + fx, cy + fy, BASE + z1, False)
        for fx, fy in ((h, h), (h, -h), (-h, h), (-h, -h)):
            finial(cx + fx, cy + fy, BASE + z1, h >= 4)

    # peak: one big bell finial tapering to a 2 px point
    zc = BASE + H - ch
    for t in range(crown_h):
        f = t / crown_h
        r = crown if f < 0.25 else max(1, round(crown * (1 - f) / 0.75 - 0.2))
        put((ZZ == zc + t) & (ax < r) & (ay < r), 4 if f >= 0.75 else 3)


ALL4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
BY, FY, AYP = -25, 41, 14            # back row, front row, end temples (y)
FX, APX = 52, 61                     # row x offset, end temples x
# Shiva: 46 px plinth, 66 tall
temple(0, BY, 66, [23, 21, 19], 9, 13, 3, 6, 14, [12, 9, 6, 3], 3, ALL4, 5, 4)
# Brahma (-x), Vishnu (+x): 34 px, 50 tall
for x in (-FX, FX):
    temple(x, BY, 50, [17, 15, 14], 7, 10, 3, 4, 11, [9, 7, 5, 3], 2, ALL4, 4, 3)
# Hamsa, Nandi, Garuda: 24 px, 34 tall, facing the big temples
for x in (-FX, 0, FX):
    temple(x, FY, 34, [12, 11], 5, 7, 2, 3, 8, [6, 4, 2], 2, [(0, -1)], 3, 2)
# Apit temples at the ends of the gap: 16 px, 24 tall, facing the centre
for x in (-APX, APX):
    temple(x, AYP, 24, [8, 7], 3, 5, 1, 2, 6, [4, 3, 2], 1, [(-int(np.sign(x)), 0)], 2, 2)

# ---------------- colour by exposure ----------------
def shift(a, dz=0, dy=0, dx=0):
    out = np.zeros_like(a)
    src = a[max(0, dz):NZ + min(0, dz), max(0, dy):W + min(0, dy), max(0, dx):W + min(0, dx)]
    out[max(0, -dz):NZ - max(0, dz), max(0, -dy):W - max(0, dy), max(0, -dx):W - max(0, dx)] = src
    return out          # out[z, y, x] = a[z + dz, y + dy, x + dx]


top_face = S & ~shift(S, dz=1)
G = np.zeros((NZ, W, W), np.uint8)
stone = S & np.isin(MAT, (3, 4, 6))
G[stone] = 5
lit = np.zeros_like(S)
recess = np.zeros_like(S)
for ddy, ddx, is_lit in ((1, 0, True), (0, -1, True), (-1, 0, False), (0, 1, False)):
    open_ = ~shift(S, dy=ddy, dx=ddx)
    if is_lit:
        lit |= open_
    recess |= open_ & shift(S, dz=1, dy=ddy, dx=ddx)          # overhang right above that face
G[stone & lit] = 4
G[stone & recess & ~top_face] = 6
G[stone & top_face] = 3
G[S & (MAT == 4)] = 3
G[S & (MAT == 6)] = 3
G[S & (MAT == 5)] = 7
pav = S & (MAT == 2)
G[pav] = 5
G[pav & lit] = 4
G[pav & top_face] = 2
G[S & (MAT == 1)] = 1

# ---------------- write the strip ----------------
used = sorted(set(np.unique(G)) - {0})
strip = np.zeros((W, W * NZ, 4), np.uint8)
for z in range(NZ):
    rgba = np.zeros((W, W, 4), np.uint8)
    for k in used:
        rgba[G[z] == k] = (*PAL[k], 255)
    strip[:, z * W:(z + 1) * W] = rgba
Image.fromarray(strip).save(HERE / "slices.png")
zs = np.where(G.any(axis=(1, 2)))[0]
print(f"grid {W}x{W}x{NZ}, filled z {zs.min()}..{zs.max()}, colours {len(used)}, voxels {int((G > 0).sum())}")
