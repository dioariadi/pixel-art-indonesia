"""Lawang Sewu (Semarang, 1907) voxel model, built directly as a material volume and written as a
sprite-stack slice strip (ground slice on the left). Method copied from ../gedung-sate/build.py.

Axes: x right, y toward the viewer at turn 0 of the raw strip, z up. Grid 121 x 121 x 60, axis (60, 60).
Plan: L of two wings (south wing along x, east wing along y) meeting at the (+x, +y) corner, which is
chamfered on the line x + y = K into the entrance facade, flanked by two octagonal towers with dark
bell domes. The building is symmetric about the diagonal x = y, which passes through the axis.
The renderer adds a 45 degree turn offset so frame 0 looks straight at the entrance facade.
Courtyard side: a tall stair block with a big arched window and two small red-domed turrets.
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
W, NZ = 121, 60
C = (W - 1) // 2                     # 60: rotation axis
O = -8                               # shift along x and y: footprint centroid sits ~3 px from the axis toward the towers

PAL = {
    1: (110, 158, 86),    # lawn
    2: (200, 194, 180),   # paving
    3: (246, 240, 224),   # wall cream (faces looking along y)
    4: (214, 206, 186),   # wall cream shade (faces looking along x, tops)
    5: (40, 34, 40),      # openings
    6: (128, 48, 34),     # roof dark (eaves, gable edge)
    7: (196, 88, 52),     # roof mid (slopes facing y)
    8: (226, 124, 74),    # roof light (slopes facing x)
    9: (58, 62, 72),      # dome dark
    10: (110, 116, 128),  # dome light
    11: (204, 128, 60),   # orange plinth / string course
}

ZZ, YY, XX = np.mgrid[0:NZ, 0:W, 0:W]
S = np.zeros((NZ, W, W), bool)
MAT = np.zeros((NZ, W, W), np.uint8)   # 1 lawn, 2 paving, 3 wall, 6/7/8 roof, 9 dome, 10 finial
REG = np.zeros((NZ, W, W), np.uint8)   # 1 wing, 2 entrance, 3 tower, 4 stair block, 5 dormer/vent

Y2, X2 = np.mgrid[0:W, 0:W]


def put(mask, mat, reg=0):
    S[mask] = True
    MAT[mask] = mat
    REG[mask] = reg


def col(mask2d, z0, z1):
    return mask2d[None] & (ZZ >= z0) & (ZZ <= z1)


# ---------------- plan (2D) ----------------
X1 = 100 + O                         # outer faces of the wings
X0 = 6                               # far ends of the wings
D = 16                               # wing depth
K = 2 * X1 - 24                      # chamfer line x + y = K (entrance facade)
SUM, DIF = X2 + Y2, X2 - Y2
wing_s = (X2 >= X0) & (X2 <= X1) & (Y2 >= X1 - D + 1) & (Y2 <= X1)
wing_e = (Y2 >= X0) & (Y2 <= X1) & (X2 >= X1 - D + 1) & (X2 <= X1)
WINGS = (wing_s | wing_e) & (SUM <= K)
STRIP = (SUM >= K - 16) & (SUM <= K) & (np.abs(DIF) <= 24)          # entrance block, diagonal
STAIR = (SUM >= K - 30) & (SUM < K - 16) & (np.abs(DIF) <= 10)      # courtyard stair block
TOWERS = [((K + 20) // 2, (K - 20) // 2), ((K - 20) // 2, (K + 20) // 2)]
TURRETS = [((K - 24 + 12) // 2, (K - 24 - 12) // 2), ((K - 24 - 12) // 2, (K - 24 + 12) // 2)]


def octagon(cx, cy, r):
    dx, dy = np.abs(X2 - cx), np.abs(Y2 - cy)
    return (np.maximum(dx, dy) <= r) & (dx + dy <= r + r // 2 + (r > 3))


# ---------------- ground ----------------
plate = (X2 >= 2) & (X2 <= W - 3) & (Y2 >= 2) & (Y2 <= W - 3)
put(col(plate, 0, 0), 1)
fore = plate & (SUM >= K - 8) & ~WINGS
walk = plate & ~WINGS & (SUM > 2 * (X1 - D)) & ((X2 > X1) | (Y2 > X1)) & ((X2 <= X1 + 4) | (Y2 <= X1 + 4))
put(col(fore | walk, 0, 0), 2)
# courtyard path from the stair block toward the inner corner of the plate
path = plate & (np.abs(DIF) <= 3) & (SUM < K - 30)
put(col(path, 0, 0), 2)

# ---------------- wings ----------------
EAVE = 21
put(col(WINGS, 1, EAVE - 1), 3, 1)

# roof: chebyshev distance field over the wing footprint + 2 px overhang gives hips and valleys
F = WINGS.copy()
for _ in range(2):
    p = np.pad(F, 1)
    F = F | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] | p[2:, 2:] | p[:-2, 2:] | p[2:, :-2]
F &= SUM <= K
dist = np.zeros((W, W), int)
cur = F.copy()
k = 0
while cur.any():
    k += 1
    dist[cur] = k
    p = np.pad(cur, 1)
    cur = cur & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:] & p[:-2, :-2] & p[2:, 2:] & p[:-2, 2:] & p[2:, :-2]


def run_dist(mask, axis):
    """Distance to the nearest outside cell along one axis."""
    out = np.zeros((W, W), int)
    m = mask if axis == 1 else mask.T
    o = out if axis == 1 else out.T
    for r in range(W):
        row = m[r]
        last = -999
        left = np.zeros(W, int)
        for i in range(W):
            if not row[i]:
                last = i
            left[i] = i - last
        last = 999 + W
        for i in range(W - 1, -1, -1):
            if not row[i]:
                last = i
            o[r, i] = min(left[i], last - i) if row[i] else 0
    return out


dx_run, dy_run = run_dist(F, 1), run_dist(F, 0)
rtop = EAVE + np.floor((dist - 1) * 1.25).astype(int)
for z in range(EAVE, rtop.max() + 1):
    m = F & (rtop >= z)
    tone = np.where(dist == 1, 6, np.where(dx_run < dy_run, 8, 7))
    for t in (6, 7, 8):
        put(m[None] & (tone == t)[None] & (ZZ == z), t, 1)

# dormers on the two street-side slopes (courtyard ones read as speckle at this size), ridge ventilators
def dormer_x(cx, face_y, sgn):
    """Dormer on a slope facing +y (sgn 1) or -y (sgn -1), front wall at y = face_y."""
    ys = np.arange(face_y - 4 * sgn, face_y + sgn, sgn)
    m2 = (np.abs(X2 - cx) <= 2) & np.isin(Y2, ys)
    put(col(m2, EAVE + 1, EAVE + 4), 3, 5)
    put(col(m2 | ((np.abs(X2 - cx) <= 3) & (Y2 == face_y + sgn)), EAVE + 5, EAVE + 5), 7, 5)
    put(col((np.abs(X2 - cx) <= 1) & np.isin(Y2, ys), EAVE + 6, EAVE + 6), 7, 5)


def dormer_y(cy, face_x, sgn):
    xs = np.arange(face_x - 4 * sgn, face_x + sgn, sgn)
    m2 = (np.abs(Y2 - cy) <= 2) & np.isin(X2, xs)
    put(col(m2, EAVE + 1, EAVE + 4), 3, 5)
    put(col(m2 | ((np.abs(Y2 - cy) <= 3) & (X2 == face_x + sgn)), EAVE + 5, EAVE + 5), 8, 5)
    put(col((np.abs(Y2 - cy) <= 1) & np.isin(X2, xs), EAVE + 6, EAVE + 6), 8, 5)


DORM = list(range(X0 + 8, X1 - 22, 12))
for c in DORM:
    dormer_x(c, X1 - 1, 1)               # street side of the south wing
    dormer_y(c, X1 - 1, 1)
ridge = X1 - D // 2                      # ridge rows X1-8, X1-7
for c in DORM[::2]:
    c2 = c + 6
    vt = rtop[ridge, c2] + 1
    v = (np.abs(X2 - c2) <= 1) & (Y2 >= ridge - 2) & (Y2 <= ridge + 1)
    put(col(v, vt, vt + 1), 3, 5)
    put(col((np.abs(X2 - c2) <= 2) & (Y2 >= ridge - 3) & (Y2 <= ridge + 2), vt + 2, vt + 2), 6, 5)
    v = (np.abs(Y2 - c2) <= 1) & (X2 >= ridge - 2) & (X2 <= ridge + 1)
    put(col(v, vt, vt + 1), 3, 5)
    put(col((np.abs(Y2 - c2) <= 2) & (X2 >= ridge - 3) & (X2 <= ridge + 2), vt + 2, vt + 2), 6, 5)

# ---------------- entrance block with gable ----------------
EW = 32                                    # entrance walls z 1..32, gable above
gtop = EW + np.maximum(0, 12 - np.abs(DIF) // 2)
for z in range(1, gtop.max() + 1):
    m = STRIP & (gtop >= z)
    roofband = STRIP & (z > EW) & (gtop - z <= 1)       # top 2 layers of the gable roof
    put(m[None] & ~roofband[None] & (ZZ == z), 3, 2)
    put((m & roofband)[None] & (ZZ == z), 7, 2)
# gable roof eave rim on the facade edge
rim = STRIP & (SUM >= K - 1) & (gtop > EW)
for z in range(EW + 1, gtop.max() + 1):
    put((rim & (gtop - z <= 1))[None] & (ZZ == z), 6, 2)

# ---------------- stair block (courtyard side) and turrets ----------------
put(col(STAIR, 1, 36), 3, 4)
put(col(STAIR & ~((SUM >= K - 28) & (SUM < K - 17) & (np.abs(DIF) <= 8)), 37, 37), 3, 4)   # parapet
for tx, ty in TURRETS:
    put(col(octagon(tx, ty, 2), 1, 34), 3, 4)
    for z, r in ((35, 3), (36, 3), (37, 2), (38, 1), (39, 0)):
        m2 = octagon(tx, ty, r) if r else ((X2 == tx) & (Y2 == ty))
        put(col(m2, z, z), 7, 4)

# ---------------- towers ----------------
TW = 45
for tx, ty in TOWERS:
    put(col(octagon(tx, ty, 5), 1, TW), 3, 3)
    put(col(octagon(tx, ty, 6), TW + 1, TW + 1), 3, 3)          # cornice
    for z, r in zip(range(TW + 2, TW + 10), (5, 5, 5, 4, 4, 3, 2, 1)):
        put(col(octagon(tx, ty, r), z, z), 9, 3)
    put(col((np.abs(X2 - tx) + np.abs(Y2 - ty) <= 1), TW + 10, TW + 12), 10, 3)
    put(col((X2 == tx) & (Y2 == ty), TW + 13, TW + 13), 10, 3)

# ---------------- colour by exposure ----------------
G = np.zeros((NZ, W, W), np.uint8)


def empty_shift(axis, d):
    e = np.ones_like(S)
    sl_to = [slice(None)] * 3
    sl_from = [slice(None)] * 3
    if d > 0:
        sl_to[axis], sl_from[axis] = slice(None, -1), slice(1, None)
    else:
        sl_to[axis], sl_from[axis] = slice(1, None), slice(None, -1)
    e[tuple(sl_to)] = ~S[tuple(sl_from)]
    return e


EX = {"+x": S & empty_shift(2, 1), "-x": S & empty_shift(2, -1),
      "+y": S & empty_shift(1, 1), "-y": S & empty_shift(1, -1)}
above = np.zeros_like(S)
above[:-1] = S[1:]
top = S & ~above

wall = S & (MAT == 3)
G[wall] = 3
G[wall & (EX["+x"] | EX["-x"])] = 4
G[wall & top] = 4
for t in (6, 7, 8):
    G[S & (MAT == t)] = t
dome = S & (MAT == 9)
G[dome] = 9
G[dome & (EX["+x"] | EX["-x"] | top)] = 10
G[S & (MAT == 10)] = 10
G[S & (MAT == 1)] = 1
G[S & (MAT == 2)] = 2
# orange plinth and string course on all walls
G[wall & ((ZZ == 1) | (ZZ == 9)) & (REG != 5)] = 11

# ---------------- openings (painted 2 voxels deep so squash never skips a row) ----------------
def deep(ex, axis, d):
    """Exposed voxels plus the voxel just behind them."""
    b = np.zeros_like(ex)
    if axis == 2:
        if d > 0:
            b[:, :, :-1] = ex[:, :, 1:]
        else:
            b[:, :, 1:] = ex[:, :, :-1]
    else:
        if d > 0:
            b[:, :-1] = ex[:, 1:]
        else:
            b[:, 1:] = ex[:, :-1]
    return (ex | b) & S


FACES = {k: deep(EX[k], 2 if "x" in k else 1, 1 if k[0] == "+" else -1) for k in EX}
FY = FACES["+y"] | FACES["-y"]
FX = FACES["+x"] | FACES["-x"]
DARK = np.zeros_like(S)

# wings: the "thousand doors" arcade, 3 px arches with 2 px piers, two storeys
keep_off = ~(col(STRIP | STAIR, 0, NZ - 1))
for zz in [(2, 6, 7), (12, 16, 17)]:      # ground arcade / upper gallery, solid bands between
    z0, z1, ztop = zz
    for faces, along in ((FY, XX - X0), (FX, YY - X0)):
        ph = along % 5
        base = faces & wall & (REG == 1) & keep_off
        DARK |= base & (ZZ >= z0) & (ZZ <= z1) & np.isin(ph, (1, 2, 3))
        DARK |= base & (ZZ == ztop) & (ph == 2)

# entrance facade (chamfer face, x + y >= K - 1): three storeys of tall arches
AD = np.abs(XX - YY)
facade = wall & (REG == 2) & ((XX + YY) >= K - 1)
for z0, z1 in ((2, 9), (12, 19), (22, 29)):
    DARK |= facade & (ZZ >= z0) & (ZZ < z1) & ((AD <= 2) | ((AD >= 8) & (AD <= 12)))
    DARK |= facade & (ZZ == z1) & ((AD <= 1) | ((AD >= 9) & (AD <= 11)))
DARK |= facade & (ZZ >= 37) & (ZZ <= 39) & (AD <= 2)                       # round gable window
DARK |= facade & (ZZ == 36) & (AD <= 1)
DARK |= facade & (ZZ == 40) & (AD <= 1)

# stair block: tall arched window toward the courtyard, door below
sface = wall & (REG == 4) & ((XX + YY) <= K - 29) & (AD <= 10)
DARK |= sface & (ZZ >= 13) & (ZZ <= 30) & (AD <= 2)
DARK |= sface & (ZZ == 31) & (AD <= 1)
DARK |= sface & (ZZ >= 2) & (ZZ <= 8) & (AD <= 2)
DARK |= sface & (ZZ >= 13) & (ZZ <= 28) & (AD >= 7) & (AD <= 9)

# towers: windows centred on each face of the octagon
for tx, ty in TOWERS:
    adx, ady = np.abs(XX - tx), np.abs(YY - ty)
    shell = wall & (REG == 3) & (np.maximum(adx, ady) >= 4) & (ZZ <= TW)
    onface = (np.minimum(adx, ady) <= 1) | ((np.minimum(adx, ady) >= 3) & (np.abs(adx - ady) <= 1))
    for z0, z1 in ((3, 8), (13, 18), (23, 28), (34, 41)):
        DARK |= shell & onface & (ZZ >= z0) & (ZZ <= z1)

# dormer windows and ventilator slats
dorm = wall & (REG == 5)
DARK |= dorm & (FACES["+y"] | FACES["+x"]) & (ZZ >= EAVE + 2) & (ZZ <= EAVE + 3) & ~top

G[DARK & S & (MAT == 3)] = 5

# ---------------- checks ----------------
assert (G[0] > 0).sum() == plate.sum(), "ground plate has holes"
assert (S == S.transpose(0, 2, 1)).all() and ((G == 5) == (G == 5).transpose(0, 2, 1)).all(), "shape and openings must be symmetric about the diagonal x = y"

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
fp = G[1:].any(axis=0)
yy, xx = np.nonzero(fp)
print(f"grid {W}x{W}x{NZ}, filled z {zs.min()}..{zs.max()}, colours {len(used)}, voxels {int((G > 0).sum())}, "
      f"building footprint centroid ({xx.mean():.1f},{yy.mean():.1f}) bbox x {xx.min()}..{xx.max()} vs axis {C}")
