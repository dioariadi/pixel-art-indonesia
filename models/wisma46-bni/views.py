"""Draw the 4-view sheet (TOP, SIDE, FRONT, BACK) of Wisma 46 at native pixel size.

Axes: x = across the broad faces (W), y = along the narrow faces (L, y = L-1 is the FRONT),
z = up (H).
- FRONT and BACK (broad faces): white window-grid shaft, full width, podium to ~68% height.
- SIDE (narrow faces): white shaft with a narrow blue glass strip in the middle third,
  crossed by a darker blue band.
- Above the shaft: a blue glass sail. From the broad faces its top edge is one curve that
  climbs from the low shoulder (x = 2) to the peak at the opposite edge (x = 31), where the
  thin white spire stands. From the narrow faces it is a flat-topped straight-sided block.
Output: views.png (8x, flat white, views centred on one row).
"""
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).with_name("views.png")
W, L, H = 34, 22, 114
SCALE, GAP = 8, 48

# ---- palette (16 max, shading drawn in; carve with --shade 0) ----
INK = (24, 28, 48)
CREAM_HI = (244, 236, 214)
CREAM = (222, 212, 188)
CREAM_LO = (186, 172, 150)
WINDOW = (92, 104, 132)
GLASS_HI = (120, 214, 232)
GLASS = (52, 150, 204)
GLASS_LO = (28, 88, 156)
BAND = (22, 54, 112)
FLOOR = (160, 226, 240)   # light floor lines across the glass top
FRAME = (206, 214, 228)
STONE = (150, 146, 140)
STONE_LO = (104, 100, 104)

# ---- geometry ----
PODIUM_X, PODIUM_Y, PODIUM_Z = (0, 33), (0, 21), 6
SHAFT_X, SHAFT_Y = (2, 31), (2, 19)
SHAFT_TOP = 77                      # ~68% of 114
STRIP_Y = (8, 13)                   # middle third of the narrow face
BAND_Z = (46, 52)
SAIL_Y = (3, 18)                    # a little shallower than the shaft
PEAK = 104
SPIRE_X, SPIRE_Y = (30, 31), (10, 11)
SPIRE_TOP = H - 1


def sail_top(x: int) -> int:
    """Top edge of the sail seen from the FRONT: rounded shoulder at x = 2, climbing to the peak at x = 31."""
    t = (x - SHAFT_X[0]) / (SHAFT_X[1] - SHAFT_X[0])
    t = min(max(t, 0.0), 1.0)
    rise = 0.55 * np.sqrt(1 - (1 - t) ** 2) + 0.45 * t ** 2
    return int(round(SHAFT_TOP + 1 + (PEAK - SHAFT_TOP - 1) * rise))


SAIL_CY = (SAIL_Y[0] + SAIL_Y[1]) / 2          # 10.5, spire y 10..11 sits on it
SAIL_HALF = (SAIL_Y[1] - SAIL_Y[0] + 1) / 2     # 8


def sail_half(z: int) -> float:
    """Half-depth of the sail seen from the narrow face: full at the shaft top, narrowing with
    slightly concave (tent-like) edges to the spire's half-width (1) at the peak."""
    s = (z - SHAFT_TOP - 1) / (PEAK - SHAFT_TOP - 1)
    return 1 + (SAIL_HALF - 1) * (1 - s) ** 1.5


def in_taper(y: int, z: int) -> bool:
    return abs(y - SAIL_CY) <= sail_half(z)


def surface(x: int, y: int) -> int:
    """Height of the sail's top surface at (x, y): min of the broad-face curve and the taper."""
    zt = max((z for z in range(SHAFT_TOP + 1, PEAK + 1) if in_taper(y, z)), default=SHAFT_TOP)
    return min(sail_top(x), zt)


def glass_colour(z: int, base):
    """Glass floors: a lighter line every 4 storeys above the shaft."""
    return FLOOR if (z - SHAFT_TOP) % 4 == 0 else base


def blank(h, w):
    return np.zeros((h, w, 4), np.uint8)


def put(a, y, x, c):
    a[y, x, :3] = c
    a[y, x, 3] = 255


def outline(a, keep=None):
    """Dark 1px outline on the outer silhouette pixels, skipping `keep` (the spire)."""
    m = a[..., 3] > 0
    p = np.pad(m, 1)
    edge = m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    if keep is not None:
        edge &= ~keep
    a[edge, :3] = INK
    return a


def row(z):
    return H - 1 - z


def podium(z, i):
    return STONE_LO if z in (1, 2) and i % 3 else STONE


def draw_side():
    """Narrow face, indexed by y (back at col 0); flipped at the end so the FRONT points LEFT."""
    a = blank(H, L)
    spire = np.zeros((H, L), bool)
    for y in range(L):
        for z in range(H):
            c = None
            if z <= PODIUM_Z and PODIUM_Y[0] <= y <= PODIUM_Y[1]:
                c = podium(z, y)
            elif SHAFT_Y[0] <= y <= SHAFT_Y[1] and z <= SHAFT_TOP:
                if STRIP_Y[0] <= y <= STRIP_Y[1]:
                    if BAND_Z[0] <= z <= BAND_Z[1]:
                        c = BAND
                    elif y in STRIP_Y:
                        c = GLASS_LO
                    elif y == STRIP_Y[0] + 2:
                        c = GLASS_HI
                    else:
                        c = GLASS
                elif z >= SHAFT_TOP - 1:
                    c = CREAM_HI
                elif y in (SHAFT_Y[0], SHAFT_Y[1]):
                    c = CREAM_LO
                elif y % 2 == 1 and z % 3 == 0 and z > PODIUM_Z + 1:
                    c = WINDOW
                else:
                    c = CREAM
            elif SHAFT_TOP < z <= PEAK and in_taper(y, z):
                # 2 px in from the silhouette: the outermost pixel becomes the outline and is stripped
                if not in_taper(y - 2, z) or not in_taper(y + 2, z):
                    c = FRAME                      # white nib frame on the slanted edges
                else:
                    c = glass_colour(z, GLASS)
            if c is not None:
                put(a, row(z), y, c)
    for z in range(PEAK + 1, SPIRE_TOP + 1):
        for y in range(SPIRE_Y[0], SPIRE_Y[1] + 1):
            put(a, row(z), y, FRAME)
            spire[row(z), y] = True
    outline(a, spire)
    return a[:, ::-1]


def draw_front(lobby: bool = True):
    """Broad face: white grid shaft full width, sail climbing to the peak on the right (x = 31)."""
    a = blank(H, W)
    spire = np.zeros((H, W), bool)
    for x in range(W):
        for z in range(H):
            c = None
            if z <= PODIUM_Z and PODIUM_X[0] <= x <= PODIUM_X[1]:
                if lobby and 13 <= x <= 20 and z <= 3:
                    c = INK if z <= 2 else STONE_LO   # lobby entrance
                else:
                    c = podium(z, x)
            elif SHAFT_X[0] <= x <= SHAFT_X[1] and z <= SHAFT_TOP:
                if z >= SHAFT_TOP - 1:
                    c = CREAM_HI
                elif x in (SHAFT_X[0], SHAFT_X[1]):
                    c = CREAM_LO
                elif x % 2 == 1 and z % 3 == 0 and z > PODIUM_Z + 1:
                    c = WINDOW
                else:
                    c = CREAM_HI if x < 17 else CREAM
            elif SHAFT_X[0] <= x <= SHAFT_X[1] and SHAFT_TOP < z <= sail_top(x):
                if z >= sail_top(x) - 1:
                    c = FRAME
                elif x in (SHAFT_X[0], SHAFT_X[1]):
                    c = GLASS_LO
                else:
                    c = glass_colour(z, GLASS)
            if c is not None:
                put(a, row(z), x, c)
    for z in range(sail_top(SPIRE_X[1]) + 1, SPIRE_TOP + 1):
        for x in range(SPIRE_X[0], SPIRE_X[1] + 1):
            put(a, row(z), x, FRAME)
            spire[row(z), x] = True
    outline(a, spire)
    return a


def draw_top():
    """Top view looking down, FRONT pointing DOWN. The sail slope covers the shaft roof."""
    a = blank(L, W)
    for y in range(L):
        for x in range(W):
            c = None
            in_sail = SHAFT_X[0] <= x <= SHAFT_X[1] and SAIL_Y[0] <= y <= SAIL_Y[1] and surface(x, y) > SHAFT_TOP
            in_shaft = SHAFT_X[0] <= x <= SHAFT_X[1] and SHAFT_Y[0] <= y <= SHAFT_Y[1]
            if SPIRE_X[0] <= x <= SPIRE_X[1] and SPIRE_Y[0] <= y <= SPIRE_Y[1]:
                c = FRAME
            elif in_sail:
                # floor lines where the surface steps up across a 4-storey level
                f = (surface(x, y) - SHAFT_TOP) // 4
                crossed = any(
                    (surface(x + dx, y + dy) - SHAFT_TOP) // 4 < f
                    for dx, dy in ((-1, 0), (1, 0))  # x steps only: y steps on the taper speckle
                    if SHAFT_X[0] <= x + dx <= SHAFT_X[1] and surface(x + dx, y) > SHAFT_TOP
                )
                if crossed:
                    c = FLOOR
                else:  # no highlight stripe here: on the slope it reads as a diagonal streak
                    c = GLASS
            elif in_shaft:
                c = CREAM_HI
            elif PODIUM_X[0] <= x <= PODIUM_X[1] and PODIUM_Y[0] <= y <= PODIUM_Y[1]:
                c = STONE
            if c is not None:
                put(a, y, x, c)
    outline(a)
    return a


def main():
    front = draw_front()
    views = [draw_top(), draw_side(), front, draw_front(lobby=False)[:, ::-1].copy()]  # BACK = mirror of FRONT, no door
    big = [Image.fromarray(v, "RGBA").resize((v.shape[1] * SCALE, v.shape[0] * SCALE), Image.NEAREST) for v in views]
    sw = sum(b.width for b in big) + GAP * (len(big) + 1)
    sh = max(b.height for b in big) + GAP * 2
    sheet = Image.new("RGBA", (sw, sh), (255, 255, 255, 255))
    x = GAP
    for b in big:
        sheet.alpha_composite(b, (x, (sh - b.height) // 2))  # centred: slicer reads rows by midpoint
        x += b.width + GAP
    sheet.convert("RGB").save(OUT)
    cols = {tuple(c) for v in views for c in v[v[..., 3] > 0][:, :3]}
    print(f"saved {OUT}  top {views[0].shape[:2]} side {views[1].shape[:2]} front {views[2].shape[:2]} colours={len(cols)}")


if __name__ == "__main__":
    main()
