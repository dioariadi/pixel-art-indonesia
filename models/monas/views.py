"""Draw the 4-view sheet (TOP, SIDE, FRONT, BACK) of Monas at native pixel size, and carve it.

Monas is square in plan and four-fold symmetric, so all four elevations share one profile.
Axes: x across, y along (y = N-1 is the FRONT), z up. Bottom to top:
  plinth (stepped square) -> cawan (bowl flaring out to a wide flat terrace) -> tapering
  obelisk shaft -> small flared observation platform -> bronze pedestal -> gold flame.
Faces: FRONT/BACK are the lighter marble, both SIDE faces a step darker, so the corners read
while it turns.

Usage:
  python views.py            # writes views.png
  python views.py --carve    # also carves slices.png, check.png, viewer.html with the skill's
                             # slicer; slices from FLAME_Z0 up are clipped to a disc so the flame
                             # is round in plan while everything below stays square.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).parent
OUT = HERE / "views.png"
SLICER = Path("views_to_slices.py")  # slicer from github.com/danielwh2/sprite-stacking
N = 50            # footprint (square)
H = 126           # slices
CX = (N - 1) / 2  # 24.5
SCALE, GAP = 8, 48

# ---- palette (11 colours, shading drawn in; carve with --shade 0) ----
INK = (28, 30, 52)
MARBLE_HI = (240, 236, 226)
MARBLE = (208, 204, 196)
MARBLE_LO = (164, 162, 160)
FLOOR = (190, 184, 172)
STONE = (140, 136, 134)
STONE_LO = (106, 104, 112)
ARCH = (70, 76, 108)
GOLD_HI = (255, 222, 96)
GOLD = (232, 170, 40)
GOLD_LO = (166, 104, 30)

# ---- profile: half-width per z (square parts) ----
PLINTH = {0: 25, 1: 25, 2: 24, 3: 24, 4: 23}
BOWL_Z = (5, 20)
TERRACE_Z = (21, 22)
SHAFT_Z = (23, 103)
PLATFORM = {104: 4, 105: 5, 106: 6, 107: 7, 108: 7, 109: 5}
PEDESTAL_Z = 110
FLAME_Z0, FLAME_Z1 = 111, 125


def half_width(z: int) -> float:
    if z in PLINTH:
        return PLINTH[z]
    if BOWL_Z[0] <= z <= BOWL_Z[1]:
        t = (z - BOWL_Z[0]) / (BOWL_Z[1] - BOWL_Z[0])
        return round(11 + 11 * t ** 1.8)           # 22 px wide at the foot -> 44 under the terrace
    if TERRACE_Z[0] <= z <= TERRACE_Z[1]:
        return 23                                    # 46 px terrace
    if SHAFT_Z[0] <= z <= SHAFT_Z[1]:
        t = (z - SHAFT_Z[0]) / (SHAFT_Z[1] - SHAFT_Z[0])
        return round(5 - 2 * t)                      # 10 px -> 6 px
    if z in PLATFORM:
        return PLATFORM[z]
    if z == PEDESTAL_Z:
        return 3
    if FLAME_Z0 <= z <= FLAME_Z1:
        s = (z - FLAME_Z0) / (FLAME_Z1 - FLAME_Z0)
        if s < 0.33:
            return 3 + 2 * (s / 0.33) ** 0.6          # 6 px -> 10 px at a third of the way up
        return max(1.0, 5 * (1 - (s - 0.33) / 0.67) ** 0.85)
    return 0


def inside(i: int, z: int) -> bool:
    hw = half_width(z)
    return hw > 0 and abs(i - CX) <= hw - 0.5


def blank(h, w):
    return np.zeros((h, w, 4), np.uint8)


def put(a, r, c, col):
    a[r, c, :3] = col
    a[r, c, 3] = 255


def outline(a, keep_rows=0):
    """Dark 1px outline on the outer silhouette, except the top `keep_rows` rows (2px flame tip:
    the slicer cannot strip an outline that has no interior pixel behind it)."""
    m = a[..., 3] > 0
    p = np.pad(m, 1)
    edge = m & ~(p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    edge[:keep_rows] = False
    a[edge, :3] = INK
    return a


def elevation(light: bool):
    """One face. light=True for FRONT/BACK, False for the SIDE faces."""
    a = blank(H, N)
    hi, mid, lo = (MARBLE_HI, MARBLE, MARBLE_LO) if light else (MARBLE, MARBLE_LO, STONE)
    for z in range(H):
        r = H - 1 - z
        for i in range(N):
            if not inside(i, z):
                continue
            c = None
            if z in PLINTH:
                top_of_step = z in (1, 3, 4)
                c = (mid if top_of_step else (STONE if light else STONE_LO))
            elif BOWL_Z[0] <= z <= BOWL_Z[1]:
                # three arched openings low on the face
                for ac in (CX - 9, CX, CX + 9):
                    ad = abs(i - ac)
                    if (z <= 10 and ad <= 1.5) or (z == 11 and ad <= 0.5):  # low on the bowl
                        c = ARCH
                if c is None:
                    c = mid if z <= 15 else hi           # bowl a step lighter than the plinth, bright upper lip
            elif TERRACE_Z[0] <= z <= TERRACE_Z[1]:
                c = hi if z == TERRACE_Z[1] else mid     # thin bright rim on top of the slab
            elif SHAFT_Z[0] <= z <= SHAFT_Z[1]:
                c = hi
            elif z in PLATFORM:
                c = mid if z <= 106 else hi               # dark flared underside, bright rim
                if z == 109:
                    c = mid
            elif z == PEDESTAL_Z:
                c = GOLD_LO
            else:                                         # flame: bright left, mid, dark right
                hw = half_width(z)
                u = (i - (CX - hw + 0.5)) / max(2 * hw - 1, 1)
                if z >= FLAME_Z1 - 1:
                    c = GOLD_HI
                elif u < 0.34:
                    c = GOLD_HI if light else GOLD
                elif u < 0.7:
                    c = GOLD
                else:
                    c = GOLD_LO
            put(a, r, i, c)
    return outline(a, keep_rows=H - 1 - FLAME_Z0 - 9)


def draw_top():
    """Looking down: plinth edge, terrace rim and floor, platform rim and deck, pedestal, flame disc."""
    a = blank(N, N)
    for y in range(N):
        for x in range(N):
            d = max(abs(x - CX), abs(y - CX)) + 0.5   # chebyshev ring index, 1-based
            rr = np.hypot(x - CX, y - CX)
            if rr < 5:                                # flame, round
                c = GOLD_HI if rr < 1.6 else (GOLD if rr < 3.4 else GOLD_LO)
            elif d <= 3:
                c = GOLD_LO
            elif d <= 5:
                c = FLOOR                             # platform deck
            elif d <= 7:
                c = MARBLE_HI                         # platform rim
            elif d <= 21:
                c = FLOOR                             # terrace floor
            elif d <= 23:
                c = MARBLE_HI                         # terrace rim
            elif d <= 24:
                c = MARBLE                            # plinth step
            else:
                c = MARBLE_LO
            put(a, y, x, c)
    return outline(a)


def main():
    front = elevation(True)
    side = elevation(False)
    views = [draw_top(), side, front, front.copy()]
    big = [Image.fromarray(v, "RGBA").resize((v.shape[1] * SCALE, v.shape[0] * SCALE), Image.NEAREST) for v in views]
    sw = sum(b.width for b in big) + GAP * (len(big) + 1)
    sh = max(b.height for b in big) + GAP * 2
    sheet = Image.new("RGBA", (sw, sh), (255, 255, 255, 255))
    x = GAP
    for b in big:
        sheet.alpha_composite(b, (x, (sh - b.height) // 2))   # centred: slicer reads rows by midpoint
        x += b.width + GAP
    sheet.convert("RGB").save(OUT)
    cols = {tuple(c) for v in views for c in v[v[..., 3] > 0][:, :3]}
    print(f"saved {OUT}  top {views[0].shape[:2]} side {views[1].shape[:2]} colours={len(cols)}")


def carve():
    """Run the skill's slicer, clipping only the flame slices to a disc (rest stays square)."""
    spec = importlib.util.spec_from_file_location("v2s", SLICER)
    v2s = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2s)
    base = v2s.carve

    def carve_round_flame(top, side, front, back, shade=0.0, round_=False):
        vol = base(top, side, front, back, shade=shade, round_=False)
        L, W = vol.shape[1:3]
        yy, xx = np.mgrid[0:L, 0:W]
        for z in range(FLAME_Z0, vol.shape[0]):
            occ = vol[z, ..., 3] > 0
            if not occ.any():
                continue
            ys, xs = np.nonzero(occ.any(1))[0], np.nonzero(occ.any(0))[0]
            ry, rx = (ys[-1] - ys[0]) / 2 + 0.5, (xs[-1] - xs[0]) / 2 + 0.5
            disc = ((yy - (ys[0] + ys[-1]) / 2) / ry) ** 2 + ((xx - (xs[0] + xs[-1]) / 2) / rx) ** 2 < 1
            vol[z][~disc] = 0
        return vol

    v2s.carve = carve_round_flame
    sys.argv = ["views_to_slices.py", str(OUT), str(HERE / "slices.png"), "--slices", str(H),
                "--order", "reading", "--top-nose", "down", "--shade", "0", "--colours", "14",
                "--preview", str(HERE / "check.png"), "--html", str(HERE / "viewer.html")]
    v2s.main()


if __name__ == "__main__":
    main()
    if "--carve" in sys.argv:
        carve()
