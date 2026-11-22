"""Tugu Yogyakarta (Tugu Pal Putih / Golong Gilig) voxel model, built directly as a material volume and
written as a sprite-stack slice strip (ground slice on the left). Method follows ../gedung-sate/build.py.

Axes: x right, y toward the viewer at turn 0, z up. Grid 57 x 57 x NZ, odd, so the turn axis (28, 28) is a
pixel centre and every square part (all odd widths) is centred exactly on it: the monument does not orbit.
Proportions read from refs/ref3.jpg (front, daylight) at ~9 photo px per slice, horizontally slimmed x0.7.
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
W = 57
NZ = 112
C = (W - 1) // 2

PAL = {
    1: (150, 150, 158),   # paving
    2: (92, 140, 78),     # planting
    3: (246, 245, 240),   # white (faces looking along y)
    4: (206, 210, 218),   # pale grey (faces looking along x, tops)
    5: (58, 54, 62),      # inscription panel
    6: (150, 104, 34),    # gold dark
    7: (214, 162, 46),    # gold mid
    8: (252, 218, 104),   # gold bright
    9: (226, 228, 234),   # step and ledge tops
}

ZZ, YY, XX = np.mgrid[0:NZ, 0:W, 0:W]
DX, DY = XX - C, YY - C
AX, AY = np.abs(DX), np.abs(DY)
S = np.zeros((NZ, W, W), bool)
MAT = np.zeros((NZ, W, W), np.uint8)       # 1 paving, 2 planting, 3 masonry, 7 gold body, 9 spiral
HALF = np.full(NZ, -1)                      # half-width of the monument per slice (for face painting)


def sq(w, z0, z1, mat=3):
    h = (w - 1) // 2
    m = (AX <= h) & (AY <= h) & (ZZ >= z0) & (ZZ <= z1)
    S[m] = True
    MAT[m] = mat
    HALF[z0:z1 + 1] = h


# ---------------- ground: round paved plate with a ring of low planting ----------------
R = np.hypot(DX, DY)
g = (R <= 28.3) & (ZZ == 0)
S[g], MAT[g] = True, 1
p = (R >= 24.6) & (R <= 27.6) & (ZZ >= 1) & (ZZ <= 2)
S[p], MAT[p] = True, 2

# ---------------- stepped base, plinth, sloped skirt ----------------
sq(35, 1, 3)
sq(31, 4, 6)
sq(27, 7, 9)
sq(23, 10, 15)              # plinth block
sq(21, 16, 16)              # sloped skirt up to the pedestal
sq(19, 17, 17)
sq(17, 18, 18)
# ---------------- pedestal with inscription panels ----------------
sq(15, 19, 37)
sq(17, 38, 38)              # moulding under the gold band
sq(17, 39, 41)              # gold scallop band (painted below)
sq(19, 42, 43)              # cornice
# ---------------- shaft, tapering 13 -> 11 -> 9 -> 7 ----------------
sq(13, 44, 47)              # shaft foot with the gold star
sq(11, 48, 61)
sq(9, 62, 74)
sq(7, 75, 83)
# ---------------- capital ----------------
sq(9, 84, 86)               # gold bracket band
sq(9, 87, 91)               # white drum with gold corner brackets
sq(11, 92, 92)              # flared cornice (gold rim)
sq(11, 93, 93)
sq(7, 94, 96)               # small crown
# ---------------- finial: twisted gold cone, 15 slices ----------------
FIN0 = 97
fin_r = [2.3, 2.3, 2.0, 1.6, 1.6, 1.5, 1.5, 1.1, 1.1, 1.0, 1.0, 0, 0, 0, 0]
for k, r in enumerate(fin_r):
    z = FIN0 + k
    m = (np.hypot(DX, DY) <= r) & (ZZ == z)
    S[m], MAT[m] = True, 9
assert FIN0 + len(fin_r) <= NZ

# ---------------- colour by exposure ----------------
G = np.zeros((NZ, W, W), np.uint8)
above = np.zeros_like(S)
above[:-1] = S[1:]
top = S & ~above
ex_x = S.copy()
ex_x[:, :, 1:-1] = S[:, :, 1:-1] & (~S[:, :, 2:] | ~S[:, :, :-2])
ex_x &= AX >= AY        # x-facing side

mas = S & (MAT == 3)
# tone by facing over the full depth, not just the outer ring: nearest sampling of a turned slice
# reaches the second ring on the risers, and a white second ring there striped the steps
G[mas] = 3
G[mas & (AX > AY)] = 4
G[mas & top] = 9
G[S & (MAT == 1)] = 1
G[S & (MAT == 2)] = 2

# face painting: on_y faces (|y| = half) run along x, on_x faces run along y; 2 voxels deep so the
# squashed y faces never skip a row
HZ = HALF[:, None, None]
on_y = (AY >= HZ - 1) & (AY <= HZ) & (AX <= HZ) & (HZ >= 0)
on_x = (AX >= HZ - 1) & (AX <= HZ) & (AY <= HZ) & (HZ >= 0)


def face(z0, z1, cond, col_y, col_x=None):
    """Paint cond(along, z) on all four faces; col_y on y faces, col_x (default darker) on x faces."""
    col_x = col_x if col_x is not None else col_y
    zr = (ZZ >= z0) & (ZZ <= z1) & mas
    my = zr & on_y & cond(AX, ZZ)
    mx = zr & on_x & cond(AY, ZZ)
    G[my] = col_y
    G[mx & ~my] = col_x
    return my | mx


# pedestal panels: gold frame (|a| <= 5, z 22..34), dark inscription inside (|a| <= 3, z 24..32)
face(22, 34, lambda a, z: a <= 5, 7, 6)
face(24, 32, lambda a, z: a <= 3, 5, 5)
face(22, 22, lambda a, z: a <= 2, 8, 7)      # bright crest on the frame top
# gold scallop band: alternating bright bumps over a mid ground
face(39, 41, lambda a, z: a >= 0, 7, 6)
face(40, 40, lambda a, z: (a % 3) == 0, 8, 7)
# shaft foot: gold star on each face, gold corner arches
face(44, 47, lambda a, z: ((a == 0) & (z >= 45) & (z <= 47)) | ((a <= 1) & (z == 46)), 8, 7)
face(44, 44, lambda a, z: a >= 5, 7, 6)
# shaft: gold centre line with a cross motif partway up and a lozenge higher up
face(49, 83, lambda a, z: a == 0, 7, 6)
face(56, 58, lambda a, z: (a <= 1) & (z == 57), 8, 7)
face(66, 68, lambda a, z: ((a <= 2) & (z == 67)) | ((a <= 1) & (z != 67)), 8, 7)
face(78, 79, lambda a, z: a <= 1, 8, 7)
# shaft corners: thin gold edge trim on the upper shaft
face(75, 83, lambda a, z: a == 3, 6, 6)
# capital: gold bracket band, corner brackets, gold rim on the cornice and crown
face(84, 86, lambda a, z: a >= 0, 7, 6)
face(85, 85, lambda a, z: (a % 2) == 0, 8, 7)
face(87, 91, lambda a, z: a >= 3, 7, 6)
face(92, 92, lambda a, z: a >= 0, 7, 6)
G[mas & top & (ZZ == 92)] = 8
face(96, 96, lambda a, z: a >= 0, 7, 6)
G[mas & top & (ZZ == 96)] = 8

# finial spiral: two-start helix, bright vs mid by angle sector winding with height; dark on the far side
ang = np.arctan2(DY, DX) / (2 * np.pi)
spiral = np.floor((ang * 2 + (ZZ - FIN0) / 4.0) % 2).astype(int)
fin = S & (MAT == 9)
G[fin] = np.where(spiral[fin] == 0, 8, 7)
axis = fin & (AX == 0) & (AY == 0)
G[axis] = np.where(((ZZ[axis] - FIN0) // 2) % 2 == 0, 8, 7)

# ---------------- checks ----------------
assert G[FIN0 + len(fin_r) - 1, C, C] in (7, 8), "finial tip must be the highest point"
assert not S[FIN0 + len(fin_r):].any()
for z in range(1, NZ):     # every slice symmetric under 90 degree turn: shaft sits on the axis
    assert (S[z] == np.rot90(S[z])).all(), f"slice {z} not square-symmetric"

# ---------------- write the strip ----------------
used = sorted(set(np.unique(G)) - {0})
assert not (S & (G == 0)).any(), "uncoloured voxel"
strip = np.zeros((W, W * NZ, 4), np.uint8)
for z in range(NZ):
    rgba = np.zeros((W, W, 4), np.uint8)
    for k in used:
        rgba[G[z] == k] = (*PAL[k], 255)
    strip[:, z * W:(z + 1) * W] = rgba
Image.fromarray(strip).save(HERE / "slices.png")
zs = np.where(G.any(axis=(1, 2)))[0]
print(f"grid {W}x{W}x{NZ}, filled z {zs.min()}..{zs.max()}, colours {len(used)}, voxels {int((G > 0).sum())}")
