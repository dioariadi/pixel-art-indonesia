"""Candi Borobudur voxel model, built directly as a material volume and written as a
sprite-stack slice strip (ground slice on the left). No 4-view carve: the stupa rings need
real gaps between small stupas, which a visual hull cannot produce.

Axes: x right, y toward the viewer at turn 0, z up. The grid is even-sized, so the turn axis
sits on a pixel corner; dx, dy below are half-integers and the stupa centres snap to whole
half-integers (pixel centres) so every small stupa gets an odd, symmetric footprint with a 1 px tip.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
W = 120     # slice size (x and y): 114 px temple + ground margin
NZ = 56     # slices
CX = CY = (W - 1) / 2

PAL = {
    1: (112, 146, 92),    # ground green
    2: (60, 61, 66),      # wall dark
    3: (86, 86, 90),      # wall mid (cornice band)
    4: (122, 120, 117),   # terrace top
    5: (150, 147, 141),   # terrace top light (stair treads, round-terrace rims)
    6: (38, 38, 43),      # niche dark
    7: (86, 102, 78),     # moss
    8: (158, 153, 144),   # stupa body
    9: (198, 189, 170),   # stupa top, warm highlight
    10: (112, 109, 105),  # stupa shade
}

ZZ, YY, XX = np.mgrid[0:NZ, 0:W, 0:W]
DX = (XX - CX).astype(float)
DY = (YY - CY).astype(float)
AX, AY = np.abs(DX), np.abs(DY)
R = np.hypot(DX, DY)

S = np.zeros((NZ, W, W), bool)          # solid
MAT = np.zeros((NZ, W, W), np.uint8)    # 1 ground, 2 terrace, 3 balustrade, 4 stair, 5 stupa, 6 gate

# ---------------- plan helpers ----------------
def redent(h, step=4):
    """Square of half-size h whose corners step inward twice (zigzag)."""
    return ((AX <= h) & (AY <= h - 2 * step)) | ((AX <= h - step) & (AY <= h - step)) | \
           ((AX <= h - 2 * step) & (AY <= h))


def put(mask, mat):
    S[mask] = True
    MAT[mask] = mat


# ---------------- ground ----------------
put((ZZ == 0) & (AX <= 59.5) & (AY <= 59.5), 1)

# ---------------- base platform + five square terraces ----------------
SQ = [(56.5, 3), (52.5, 7), (48.5, 11), (44.5, 15), (41.5, 19), (38.5, 23)]   # (half, top z)
for h, top in SQ:
    put(redent(h) & (ZZ >= 1) & (ZZ <= top), 2)
# balustrade: 2 px wide, 2 px tall wall on the outer edge of each terrace top (not the base)
for h, top in SQ[1:]:
    ring = redent(h) & ~redent(h - 2)
    put(ring & (ZZ >= top + 1) & (ZZ <= top + 2), 3)

# ---------------- stairways on the four sides ----------------
# along = distance from centre along the stair axis, across = sideways offset
TOPSQ = SQ[-1][0]
XP = [h for h, _ in SQ[::-1]] + [58.5]
FP = [t for _, t in SQ[::-1]] + [1]
for along, across in ((AY, AX), (AX, AY)):
    strip = (across <= 2.5) & (along >= TOPSQ - 2) & (along <= 58.5)
    hz = np.floor(np.interp(along, XP, FP))                  # lands on every terrace top
    cut = strip & (ZZ > hz) & (ZZ <= 25)
    S[cut] = False
    MAT[cut] = 0
    put(strip & (ZZ >= 1) & (ZZ <= hz), 4)
    # gateway at each terrace edge: posts either side + lintel, 2 px deep
    for h, top in SQ[1:]:
        band = (along <= h) & (along >= h - 1.5)
        posts = band & (across >= 2.5) & (across <= 4.5) & (ZZ >= top + 1) & (ZZ <= top + 4)
        lintel = band & (across <= 4.5) & (ZZ >= top + 4) & (ZZ <= top + 5)
        put(posts | lintel, 6)

# ---------------- three round terraces ----------------
ROUND = [(36.5, 27), (27.5, 31), (18.5, 35)]    # (radius, top z), all start at z 24
for rad, top in ROUND:
    put((R <= rad) & (ZZ >= 24) & (ZZ <= top), 2)

# ---------------- small stupas in rings ----------------
# (ring radius, count, terrace top, angle offset in steps); each ring sits half a step off the one below
RINGS = [(32.0, 16, 27, 0.5), (23.5, 12, 31, 0.0), (14.5, 8, 35, 0.5)]
# disc radius per layer from a pixel-centre centre: 7,7 (bell foot), 5,5, 3 (shoulder), 3 (square cap), 1,1 (tip)
BELL = [3.2, 3.2, 2.3, 2.3, 1.5, 1.5, 0.0, 0.0]
SST = np.zeros_like(S)            # small-stupa voxels, with their height within the stupa
SZ = np.full(S.shape, -1, np.int8)
d_min = 99.0
for rr, n, top, off in RINGS:
    ring_c = []
    for k in range(n):
        a = 2 * math.pi * (k + off) / n
        cx, cy = math.floor(rr * math.cos(a)) + 0.5, math.floor(rr * math.sin(a)) + 0.5
        ring_c.append((cx, cy))
        d = np.hypot(DX - cx, DY - cy)
        for i, rb in enumerate(BELL):
            m = (ZZ == top + 1 + i) & (d <= rb + 1e-6)
            if i == 5:      # square cap
                m = (ZZ == top + 1 + i) & (np.abs(DX - cx) <= 1) & (np.abs(DY - cy) <= 1)
            put(m, 5)
            SST[m] = True
            SZ[m] = i
    for i in range(n):
        a, b = ring_c[i], ring_c[(i + 1) % n]
        d_min = min(d_min, math.hypot(a[0] - b[0], a[1] - b[1]))

# separation check: neighbours in a ring are 7 px wide, so centres >= 10 px apart gives >= 3 px gap
assert d_min >= 10, f"small stupas too close: {d_min:.2f}"

# ---------------- central stupa (corner-centred: 16 px bell, the biggest and highest) ----------------
put((R <= 8.6) & (ZZ >= 36) & (ZZ <= 37), 5)                  # round base, 18 px
bell_r = [7.6, 7.6, 7.5, 7.2, 6.8, 6.2, 5.4, 4.4, 3.2]         # bell, 16 px wide at the bottom
for i, rb in enumerate(bell_r):
    put((ZZ == 38 + i) & (R <= rb), 5)
put((ZZ >= 47) & (ZZ <= 48) & (AX <= 2.5) & (AY <= 2.5), 5)   # square block (harmika), 6x6
put((ZZ >= 49) & (ZZ <= 51) & (AX <= 1.5) & (AY <= 1.5), 5)   # spire, 4x4
put((ZZ >= 52) & (ZZ <= 55) & (AX <= 0.5) & (AY <= 0.5), 5)   # spire tip, 2x2

# ---------------- colour by exposure ----------------
G = np.zeros((NZ, W, W), np.uint8)
above = np.zeros_like(S)
above[:-1] = S[1:]
top_face = S & ~above
below_top = np.zeros_like(S)
below_top[:-1] = top_face[1:]

# one rule on all four sides: walls dark, a mid cornice row, floors one lighter tone, balustrade tops lightest
terr = S & np.isin(MAT, (2, 3, 4, 6))
G[terr] = 2
G[terr & below_top & (ZZ <= 25)] = 3
G[terr & top_face] = 4
G[(MAT == 3) & top_face] = 5
G[(MAT == 6) & top_face] = 5
# round terraces: dark floors so the stupas stand out, 1 px light rim
G[(MAT == 2) & top_face & (ZZ >= 26)] = 3
for rad, top in ROUND:
    G[(MAT == 2) & top_face & (ZZ == top) & (R > rad - 1.2)] = 5
# niches: alternating dark pixels along the balustrade, lower course
along_edge = np.where(AX >= AY, DY, DX)
for h, top in SQ[1:]:
    face = (MAT == 3) & (ZZ == top + 1) & ~redent(h - 1)
    G[face & ((np.floor(along_edge + 0.5).astype(int) % 3) == 0)] = 6
G[(MAT == 2) & (ZZ == 2) & ~redent(SQ[0][0] - 1) & ((np.floor(along_edge + 0.5).astype(int) % 4) == 0)] = 6
# moss in short runs low on the walls
seg = (np.floor((along_edge + 70) / 3)).astype(int)
patch = ((seg * 7 + np.where(AX >= AY, 1, 0) * 3 + (DX > 0) * 5 + (DY > 0) * 11) % 5) == 0
moss = (MAT == 2) & ~top_face & ~below_top & (ZZ >= 1) & (ZZ <= 6) & patch & ((ZZ % 4) >= 1) & ((ZZ % 4) <= 2)
G[moss] = 7
# stupas: solid tones only. lower third dark, body mid, top faces and cap/tip warm light
st = S & (MAT == 5)
G[st] = 8
G[st & top_face] = 9
G[SST & (SZ <= 1)] = 10
G[SST & (SZ >= 5)] = 9
G[st & ~SST & (ZZ <= 40)] = 10          # central stupa: base and lower bell shaded
G[st & ~SST & (ZZ <= 37) & top_face] = 10
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
print(f"grid {W}x{W}x{NZ}, filled z {zs.min()}..{zs.max()}, colours {len(used)}, "
      f"stupas {[r[1] for r in RINGS]}, min stupa spacing {d_min:.2f}, voxels {int((G > 0).sum())}")
