"""Panggung Sangga Buwana (Keraton Surakarta) voxel model, built directly as a material volume and written as a
sprite-stack slice strip (ground slice on the left). Method follows ../tugu-yogyakarta/build.py.

Axes: x right, y toward the viewer at turn 0, z up. Grid W x W x NZ with W odd, so the turn axis (C, C) is a
pixel centre. Every octagon is the true polygon |x| <= a, |y| <= a, |x| + |y| <= a*sqrt2, centred on the axis.
Faces are assigned by the largest projection on the 8 face normals; even faces white, odd faces pale grey.

Read from refs/: ref1 (top storey, gallery, roof, dormers, vane), ref3 / ref5 (colour, full upper tower),
ref2 / ref4 (whole tower over the palace roofs). The photos show open galleries on thin blue posts that
project well past the body, a concave bell-shaped shingle roof with white hip ridges and small dormers,
blue louvred windows (arched on the top storey) and a weathervane rod on the peak.
"""
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
W = 61
NZ = 106
C = (W - 1) // 2
SQ2 = np.sqrt(2.0)

PAL = {
    1: (184, 186, 192),   # paving
    2: (246, 245, 240),   # white wall (even faces), roof ridges
    3: (208, 212, 222),   # pale grey wall (odd faces)
    4: (112, 172, 214),   # light blue: shutters, railings, posts
    5: (62, 112, 164),    # deep blue: gallery and eave undersides, railing body
    6: (92, 72, 62),      # roof dark brown (odd faces)
    7: (128, 102, 84),    # roof brown (even faces)
    8: (40, 44, 58),      # dark openings, vane
    9: (228, 230, 236),   # ledge / cornice tops
}

ZZ, YY, XX = np.mgrid[0:NZ, 0:W, 0:W]
DX, DY = (XX - C).astype(float), (YY - C).astype(float)
AX, AY = np.abs(DX), np.abs(DY)
# 8 face normals, k * 45 deg
ANG = np.arange(8) * np.pi / 4
PROJ = np.stack([np.cos(a) * DX + np.sin(a) * DY for a in ANG])     # [8, z, y, x]
FACE = np.argmax(np.round(PROJ, 6), axis=0)                          # face index per voxel
DEPTH = np.max(PROJ, axis=0)                                         # distance along the face normal
SECOND = np.sort(PROJ, axis=0)[-2]
NA = ANG[FACE]
ALONG = -np.sin(NA) * DX + np.cos(NA) * DY                           # position along the face
EVEN = (FACE % 2) == 0

S = np.zeros((NZ, W, W), bool)
MAT = np.zeros((NZ, W, W), np.uint8)   # 1 paving, 2 wall, 3 roof, 4 blue, 5 deep blue, 8 dark, 9 ledge
APO = np.full(NZ, -1.0)                # wall apothem per slice (for face painting)


def octo(a):
    """True octagon of apothem a (pixel-centre test, 90 deg symmetric)."""
    return (AX <= a + 1e-6) & (AY <= a + 1e-6) & (AX + AY <= a * SQ2 + 0.5)


def put(mask, z0, z1, mat):
    m = mask & (ZZ >= z0) & (ZZ <= z1)
    S[m] = True
    MAT[m] = mat
    return m


def wall(a, z0, z1, mat=2):
    put(octo(a), z0, z1, mat)
    APO[z0:z1 + 1] = a


# ---------------- ground plate ----------------
put(octo(27), 0, 0, 1)

# ---------------- low palace pavilion ring the tower rises from ----------------
wall(19, 1, 7)
put(octo(20), 7, 7, 9)                       # cornice
for k, a in enumerate([21, 19.5, 18, 16.5, 15, 13.5]):   # hipped shingle skirt
    put(octo(a), 8 + k, 8 + k, 3)

# ---------------- tower body ----------------
wall(12, 14, 28)                             # storey 1
put(octo(13), 29, 29, 9)                     # cornice
wall(12, 30, 43)                             # storey 2 (clock storey)
# gallery 1
G1 = 44
put(octo(15), G1, G1, 5)                     # deep blue underside
put(octo(15), G1 + 1, G1 + 1, 9)             # floor
wall(11, G1 + 1, 58)                         # storey 3 inside the gallery
put(octo(15) & ~octo(13.6), G1 + 2, G1 + 4, 4)   # railing
put(octo(15) & ~octo(13.6), G1 + 4, G1 + 4, 9)   # railing top rail
# gallery 2
G2 = 59
put(octo(16), G2, G2, 5)
put(octo(16), G2 + 1, G2 + 1, 9)
wall(10, G2 + 1, 71)                         # storey 4 (top, arched windows)
put(octo(16) & ~octo(14.6), G2 + 2, G2 + 4, 4)
put(octo(16) & ~octo(14.6), G2 + 4, G2 + 4, 9)


def posts(a, z0, z1):
    """Plus-shaped blue posts at the 8 octagon corners, inside apothem a."""
    px = int(a) - 1
    py = int(np.floor(a * SQ2 + 0.5 - px)) - 1
    m = np.zeros((W, W), bool)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for (qx, qy) in ((px, py), (py, px)):
                cx, cy = C + sx * qx, C + sy * qy
                m[cy, cx - 1:cx + 2] = True
                m[cy - 1:cy + 2, cx] = True
    put(np.broadcast_to(m, (NZ, W, W)) & octo(a), z0, z1, 4)


posts(15, G1 + 2, G2 - 1)
posts(16, G2 + 2, 71)

# ---------------- eave and bell-shaped roof ----------------
EAVE = 72
put(octo(17), EAVE, EAVE, 5)                 # blue fascia / underside
ROOF0, ROOF_N = EAVE + 1, 22
for k in range(ROOF_N):
    t = k / (ROOF_N - 1)
    a = max(1.0, 17 * (1 - t) ** 1.35)       # concave: flat flare at the eave, steep near the peak
    put(octo(a), ROOF0 + k, ROOF0 + k, 3)
ROOF_TOP = ROOF0 + ROOF_N - 1

# dormers on the four axis faces, partway up the roof
DZ0 = ROOF0 + 5
for z in range(DZ0, DZ0 + 4):
    m = EVEN & (np.abs(ALONG) <= 1.6) & (DEPTH <= 12.5) & (DEPTH >= 9)
    put(m, z, z, 2)
# dormer caps
put(EVEN & (np.abs(ALONG) <= 2.1) & (DEPTH <= 13) & (DEPTH >= 8.5), DZ0 + 4, DZ0 + 4, 3)

# ---------------- weathervane rod with a cross arm ----------------
plus = (AX + AY <= 1)
VZ0 = ROOF_TOP + 1
put(plus, VZ0, VZ0 + 9, 8)
put(((AX <= 3) & (AY == 0)) | ((AY <= 3) & (AX == 0)), VZ0 + 6, VZ0 + 6, 8)
assert VZ0 + 9 < NZ

# ---------------- colour by face ----------------
G = np.zeros((NZ, W, W), np.uint8)
above = np.zeros_like(S)
above[:-1] = S[1:]
top = S & ~above

wallm = S & (MAT == 2)
G[wallm] = np.where(EVEN[wallm], 2, 3)
G[wallm & top] = 9
roofm = S & (MAT == 3)
G[roofm] = np.where(EVEN[roofm], 7, 6)
for k in (1, 4, 5, 8, 9):
    G[S & (MAT == k)] = k

# white hip ridges on the main roof: voxels where two faces tie (octagon corners)
ridge = roofm & (ZZ >= ROOF0) & (ZZ <= ROOF_TOP) & ((DEPTH - SECOND) < 0.75)
G[ridge] = 2
# dormer window
dw = wallm & (ZZ >= DZ0 + 1) & (ZZ <= DZ0 + 2) & (np.abs(ALONG) <= 0.9) & (DEPTH >= 11)
G[dw] = 8

# ---------------- windows and doors on the 8 faces ----------------
AZ = APO[:, None, None]
shell = wallm & (DEPTH >= AZ - 1.6) & (AZ > 0)


def win(z0, z1, half, col, arch_col=None):
    m = shell & (ZZ >= z0) & (ZZ <= z1) & (np.abs(ALONG) <= half)
    G[m] = col
    if arch_col is not None:
        a = shell & (ZZ == z1 + 1) & (np.abs(ALONG) <= half - 1)
        G[a] = arch_col
    return m


win(1, 4, 2.1, 4)                 # pavilion blue doors
win(5, 5, 2.1, 5)
win(17, 23, 1.6, 4)               # storey 1 shutters
win(17, 17, 1.6, 5)
win(33, 39, 1.6, 4)               # storey 2 shutters
win(33, 33, 1.6, 5)
win(G1 + 5, G1 + 10, 1.6, 4, 8)   # storey 3, behind gallery 1
win(G2 + 5, G2 + 9, 1.6, 4, 4)    # storey 4, arched
win(G2 + 5, G2 + 5, 1.6, 5)

# ---------------- checks ----------------
assert not S[VZ0 + 10:].any()
for z in range(NZ):
    assert (S[z] == np.rot90(S[z])).all(), f"slice {z} not 90 deg symmetric"

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
