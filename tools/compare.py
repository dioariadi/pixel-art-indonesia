"""Side-by-side sheet of turntable GIFs: one row per GIF, columns at 0, 45, 180 and 315 degrees.

  python compare.py v5=old/turntable.gif new=new/turntable.gif --out compare.png [--height 400]
Frames are scaled (nearest) to a common height; labels are drawn at the left.
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ANGLES = (0, 45, 180, 315)


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+", help="label=path/to/turntable.gif")
    ap.add_argument("--out", type=Path, default=Path("compare.png"))
    ap.add_argument("--height", type=int, default=0, help="common frame height (default: tallest GIF)")
    a = ap.parse_args()

    rows = []
    for it in a.items:
        label, _, path = it.partition("=")
        if not path:
            ap.error(f"expected label=path, got {it}")
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]
        n = len(frames)
        rows.append((label, [frames[round(d / 360 * n) % n] for d in ANGLES]))
    H = a.height or max(r[1][0].height for r in rows)
    scaled = []
    for label, fr in rows:
        out = []
        for f in fr:
            w = max(1, round(f.width * H / f.height))
            out.append(f.resize((w, H), Image.NEAREST))
        scaled.append((label, out))
    colw = [max(r[1][k].width for r in scaled) for k in range(len(ANGLES))]
    f_lab, f_head = font(22), font(18)
    labw = max(int(ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(l, font=f_lab)) for l, _ in scaled) + 30
    gap, head = 12, 34
    W = labw + sum(colw) + gap * (len(ANGLES) + 1)
    Ht = head + len(scaled) * (H + gap) + gap
    sheet = Image.new("RGB", (W, Ht), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    x = labw + gap
    for k, deg in enumerate(ANGLES):
        d.text((x, 8), f"{deg} deg", fill=(40, 40, 50), font=f_head)
        x += colw[k] + gap
    for r, (label, fr) in enumerate(scaled):
        y = head + r * (H + gap)
        d.text((12, y + H // 2 - 12), label, fill=(20, 20, 30), font=f_lab)
        x = labw + gap
        for k, im in enumerate(fr):
            sheet.paste(im, (x + (colw[k] - im.width) // 2, y))
            x += colw[k] + gap
    sheet.save(a.out)
    print(f"{a.out} {sheet.size[0]}x{sheet.size[1]}, {len(scaled)} rows x {len(ANGLES)} columns {ANGLES}")


if __name__ == "__main__":
    main()
