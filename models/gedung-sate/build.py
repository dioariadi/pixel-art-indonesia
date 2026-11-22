"""Gedung Sate (Bandung, 1920) voxel model, built directly as a material volume and written as a
sprite-stack slice strip (ground slice on the left). Method copied from ../borobudur/build.py.

Axes: x right (along the main wing), y toward the viewer at turn 0 (front of the building), z up.
Grid 133 x 77 x 64, odd in x and y, so the turn axis (66, 38) is a pixel centre. The tower is
centred exactly on it: it does not orbit, and the 1 px spire rod maps onto itself at every angle.
The whole building is placed around the tower (centre block 25 x 29, centred on the axis).
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
WX, WY, NZ = 133, 77, 64
CX, CY = (WX - 1) // 2, (WY - 1) // 2      # 66, 38: rotation axis = tower axis

PAL = {
    1: (104, 150, 84),    # lawn
    2: (206, 200, 186),   # paving
    3: (242, 240, 232),   # wall white (faces looking along y)
    4: (204, 206, 208),   # wall pale grey (faces looking along x, flat tops)
    5: (34, 32, 38),      # openings
    6: (50, 42, 40),      # roof dark charcoal-brown (eave rims, sides, spire rod)
    7: (86, 74, 68),      # roof light (upward faces)
    9: (240, 196, 64),    # sate ornaments
}

ZZ, YY, XX = np.mgrid[0:NZ, 0:WY, 0:WX]
DX = XX - CX
DY = YY - CY
AX, AY = np.abs(DX), np.abs(DY)

S = np.zeros((NZ, WY, WX), bool)
MAT = np.zeros((NZ, WY, WX), np.uint8)   # 1 lawn, 2 paving, 3 wall, 6 roof, 7 skirt, 9 ornament, 10 rod
RIM = np.zeros_like(S)


def box(x0, x1, y0, y1, z0, z1):
    return (XX >= x0) & (XX <= x1) & (YY >= y0) & (YY <= y1) & (ZZ >= z0) & (ZZ <= z1)


def put(mask, mat):
    S[mask] = True
    MAT[mask] = mat


def hip(x0, x1, y0, y1, z0, h, over=2):
    """Low-pitched hipped roof over wall box x0..x1, y0..y1: eave at z0, overhang `over`, h layers."""
    x0, x1, y0, y1 = x0 - over, x1 + over, y0 - over, y1 + over
    half = (min(x1 - x0, y1 - y0) + 1) / 2
    for k in range(h):
        i = int(round(k * (half - 1) / (h - 1)))
        m = box(x0 + i, x1 - i, y0 + i, y1 - i, z0 + k, z0 + k)
        put(m, 6)
        if k == 0:
            RIM[m & ~box(x0 + 1, x1 - 1, y0 + 1, y1 - 1, z0, z0)] = True


def skirt(x0, x1, y0, y1, z=7):
    """Thin roof band between the storeys: 2 px lip at z, 1 px at z+1."""
    put(box(x0 - 2, x1 + 2, y0 - 2, y1 + 2, z, z), 7)
    put(box(x0 - 1, x1 + 1, y0 - 1, y1 + 1, z + 1, z + 1), 7)


# ---------------- ground ----------------
put(box(2, 130, 0, 76, 0, 0), 1)
put(box(CX - 7, CX + 7, 57, 76, 0, 0), 2)

# ---------------- parts (wall boxes), all symmetric about x = 66 ----------------
MAIN = (8, 124, 31, 44)          # main wing: 117 long, 14 deep
PAV_L = (6, 19, 28, 47)          # end pavilions: 14 wide, 20 deep
PAV_R = (113, 126, 28, 47)
REAR_L = (30, 43, 3, 30)         # rear wings
REAR_R = (89, 102, 3, 30)
CENTRE = (CX - 12, CX + 12, CY - 14, CY + 14)   # 25 x 29, centred on the axis, front at y 52

EAVE_MAIN = 15                    # walls z 1..14
for b in (MAIN, PAV_L, PAV_R):
    put(box(*b, 1, EAVE_MAIN - 1), 3)
for b in (REAR_L, REAR_R):
    put(box(*b, 1, 11), 3)
put(box(*CENTRE, 1, 24), 3)

for b in (MAIN, PAV_L, PAV_R, REAR_L, REAR_R, CENTRE):
    skirt(*b)
# roofs: height at most half the wall height
hip(*REAR_L, 12, 4)
hip(*REAR_R, 12, 4)
hip(*MAIN, EAVE_MAIN, 5)
hip(*PAV_L, EAVE_MAIN, 5)
hip(*PAV_R, EAVE_MAIN, 5)

# ---------------- entrance porch: small pointed gable in front of the centre block ----------------
PORCH = (CX - 5, CX + 5, CENTRE[3] + 1, CENTRE[3] + 4)    # 11 wide, 4 deep, front at y 56
put(box(*PORCH, 1, 6), 3)
for k, hw in enumerate((6, 5, 4, 3, 2, 1, 0)):            # gable roof, ridge front-to-back
    m = (AX <= hw) & (YY >= PORCH[2] - 1) & (YY <= PORCH[3] + 1) & (ZZ == 7 + k)
    put(m, 6)
    if k == 0:
        RIM[m] = True


# ---------------- tower ----------------
def sq(w, z0, z1, mat):
    """Square of odd width w centred on the axis."""
    m = (AX <= (w - 1) // 2) & (AY <= (w - 1) // 2) & (ZZ >= z0) & (ZZ <= z1)
    put(m, mat)
    return m


def tier(widths, z0, upturn=True):
    """Thin low-pitched brim: first layer is the eave (dark rim), layers step inward."""
    for k, w in enumerate(widths):
        m = sq(w, z0 + k, z0 + k, 6)
        if k == 0:
            h = (w - 1) // 2
            RIM[m & ~((AX <= h - 1) & (AY <= h - 1))] = True
    if upturn:   # corners of the eave rise one layer
        h = (widths[0] - 1) // 2
        c = (AX >= h - 1) & (AX <= h) & (AY >= h - 1) & (AY <= h) & (ZZ == z0 + 1)
        put(c, 6)
        RIM[c] = True


# balustrade on the centre block top: 1 px white rim with 2x2 corner posts
x0, x1, y0, y1 = CENTRE
rim = box(x0, x1, y0, y1, 25, 25) & ~box(x0 + 1, x1 - 1, y0 + 1, y1 - 1, 25, 25)
put(rim, 3)
for px in ((x0, x0 + 1), (x1 - 1, x1)):
    for py in ((y0, y0 + 1), (y1 - 1, y1)):
        put(box(px[0], px[1], py[0], py[1], 25, 27), 3)

# bands are 5-6 slices (not 3-4): at squash 0.6 the brim overhang hides about 2 rows of the band
T = dict(base=21, t1=(29, 23), band1=17, t2=(23, 17), band2=9, t3=(15, 9, 3))
sq(T["base"], 25, 30, 3)                 # tower base block with arched openings, z 25..30
tier(T["t1"], 31)                        # tier 1: 29 wide, 2 slices, overhang 4
sq(T["band1"], 33, 38, 3)                # white band, 6 slices
tier(T["t2"], 39)                        # tier 2: 23 wide, 2 slices, overhang 3
sq(T["band2"], 41, 45, 3)                # white band, 5 slices
tier(T["t3"], 46, upturn=False)          # tier 3: small low pyramid 15 -> 3
# spire: 1 px rod z 49..63, six 3 px beads on alternate slices, bare tip
put((AX == 0) & (AY == 0) & (ZZ >= 49) & (ZZ <= 63), 10)
for z in (50, 52, 54, 56, 58, 60):
    put((AX + AY <= 1) & (ZZ == z), 9)

# ---------------- colour by exposure ----------------
G = np.zeros((NZ, WY, WX), np.uint8)


def shifted_empty(axis, d):
    e = np.ones_like(S)
    if axis == 2:
        if d > 0:
            e[:, :, :-1] = ~S[:, :, 1:]
        else:
            e[:, :, 1:] = ~S[:, :, :-1]
    else:
        if d > 0:
            e[:, :-1, :] = ~S[:, 1:, :]
        else:
            e[:, 1:, :] = ~S[:, :-1, :]
    return e


ex_x = S & (shifted_empty(2, 1) | shifted_empty(2, -1))
ex_yp = S & shifted_empty(1, 1)     # faces the viewer at turn 0
ex_yn = S & shifted_empty(1, -1)
above = np.zeros_like(S)
above[:-1] = S[1:]
top = S & ~above

wall = S & (MAT == 3)
G[wall] = 3
G[wall & ex_x] = 4
G[wall & top] = 4
roof = S & np.isin(MAT, (6, 7))
G[roof] = 7                              # one flat upward tone: no stripes between hip steps
G[roof & RIM] = 6
G[S & (MAT == 7) & (ZZ == 8)] = 7
G[S & (MAT == 10)] = 6
G[S & (MAT == 9)] = 9
G[S & (MAT == 1)] = 1
G[S & (MAT == 2)] = 2


def openings(b, face, rows, period, margin=2, phase0=0):
    """Paint dark openings on one face of wall box b. rows: {z: set of phases within period}."""
    x0, x1, y0, y1 = b
    if face in ("front", "back"):
        yf = y1 if face == "front" else y0
        ex = ex_yp if face == "front" else ex_yn
        sel = (YY == yf) & ex & (XX >= x0 + margin) & (XX <= x1 - margin)
        along = XX - (x0 + margin)
    else:
        xf = x1 if face == "right" else x0
        sel = (XX == xf) & ex_x & (YY >= y0 + margin) & (YY <= y1 - margin)
        along = YY - (y0 + margin)
    ph = (along + phase0) % period
    for z, phases in rows.items():
        G[sel & wall & (ZZ == z) & np.isin(ph, list(phases))] = 5


ARCADE = {1: {0, 1, 2}, 2: {0, 1, 2}, 3: {0, 1, 2}, 4: {0, 1, 2}, 5: {1}}   # 3 px arch, 2 px pier
UPPER_A = {10: {1, 2}, 11: {1, 2}, 12: {1, 2}}                               # windows over the arches
REAR_WIN = {2: {1, 2}, 3: {1, 2}, 4: {1, 2}, 9: {1, 2}, 10: {1, 2}}

for f in ("front", "back"):
    openings(MAIN, f, {**ARCADE, **UPPER_A}, 5, margin=1)
for b in (PAV_L, PAV_R):
    for f in ("front", "back", "left", "right"):
        openings(b, f, {**ARCADE, **UPPER_A}, 5, margin=2)
for b in (REAR_L, REAR_R):
    for f in ("front", "back", "left", "right"):
        openings(b, f, REAR_WIN, 4, margin=2)
CENTRE_UP = {10: {1, 2}, 11: {1, 2}, 12: {1, 2}}
CENTRE_TOP = {z: {1, 2} for z in range(17, 23)}
for f in ("front", "back", "left", "right"):
    openings(CENTRE, f, {**CENTRE_UP, **CENTRE_TOP}, 4, margin=2, phase0=1)
for f in ("left", "right"):
    openings(CENTRE, f, ARCADE, 5, margin=2)
# arches either side of the porch on the centre block front
blockface = (YY == CENTRE[3]) & wall & ex_yp
for z in range(1, 6):
    for cx in (-8, 8):
        hw = 1 if z < 5 else 0
        G[blockface & (ZZ == z) & (np.abs(DX - cx) <= hw)] = 5
# pointed-arch entrance on the porch front, 5 px wide, point at z 6
pface = (YY == PORCH[3]) & wall
for z, hw in ((1, 2), (2, 2), (3, 2), (4, 2), (5, 1), (6, 0)):
    G[pface & (ZZ == z) & (AX <= hw)] = 5


def ring_face(half, z0, z1, pattern):
    """Openings on all four faces of a centred square of half-size `half`; pattern(along) -> bool."""
    on_y = (AY == half) & (AX <= half - 2)
    on_x = (AX == half) & (AY <= half - 2)
    along = np.where(on_y, AX, AY)
    m = (on_y | on_x) & wall & (ZZ >= z0) & (ZZ <= z1) & pattern(along)
    G[m] = 5


# tower base: arched openings (3 px centre, 2 px pairs at 4..5), arch tops at z 30
ring_face(10, 26, 28, lambda a: (a <= 1) | ((a >= 4) & (a <= 5)))
ring_face(10, 29, 29, lambda a: a == 0)
# bands: small dark windows
ring_face(8, 35, 37, lambda a: (a % 3) == 0)
ring_face(4, 42, 44, lambda a: a == 2)

# y faces are sampled through squash < 1, so a 1-voxel-deep paint skips rows on some slices:
# push every opening on a y face one voxel deeper
dark = G == 5
fwd = np.zeros_like(dark)
fwd[:, :-1] = dark[:, 1:] & ex_yp[:, 1:]
back = np.zeros_like(dark)
back[:, 1:] = dark[:, :-1] & ex_yn[:, :-1]
G[(fwd | back) & wall] = 5

# ---------------- checks ----------------
spire = np.argwhere(MAT == 10)
assert (spire[:, 1] == CY).all() and (spire[:, 2] == CX).all(), "spire is off the turn axis"
assert G[NZ - 1, CY, CX] == 6, "spire rod must be the highest point"

# ---------------- write the strip ----------------
used = sorted(set(np.unique(G)) - {0})
strip = np.zeros((WY, WX * NZ, 4), np.uint8)
for z in range(NZ):
    rgba = np.zeros((WY, WX, 4), np.uint8)
    for k in used:
        rgba[G[z] == k] = (*PAL[k], 255)
    strip[:, z * WX:(z + 1) * WX] = rgba
Image.fromarray(strip).save(HERE / "slices.png")
zs = np.where(G.any(axis=(1, 2)))[0]
print(f"grid {WX}x{WY}x{NZ}, filled z {zs.min()}..{zs.max()}, colours {len(used)}, voxels {int((G > 0).sum())}, "
      f"tiers {T['t1'][0]}/{T['t2'][0]}/{T['t3'][0]}")
