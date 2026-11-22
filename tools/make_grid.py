"""Tile every models/*/turntable.gif into one animated grid GIF (all models turn in sync)."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
BG = (214, 230, 242)
INK = (24, 28, 48)

LABELS = {
    "borobudur": "Borobudur",
    "gedung-sate": "Gedung Sate, Bandung",
    "lawang-sewu": "Lawang Sewu, Semarang",
    "monas": "Monas, Jakarta",
    "prambanan": "Prambanan",
    "sangga-buwana-solo": "Sangga Buwana, Solo",
    "tugu-yogyakarta": "Tugu Yogyakarta",
    "wisma46-bni": "Wisma 46, Jakarta",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--cell", type=int, default=300, help="cell width and image height in px")
    ap.add_argument("--out", type=Path, default=ROOT / "all-landmarks.gif")
    args = ap.parse_args()

    gifs = sorted((ROOT / "models").glob("*/turntable.gif"))
    ims = [Image.open(g) for g in gifs]
    n = min(im.n_frames for im in ims)
    duration = ims[0].info.get("duration", 80)
    cols = args.cols
    rows = -(-len(ims) // cols)
    cell, label_h = args.cell, 34
    font = ImageFont.load_default(size=18)
    W, H = cols * cell, rows * (cell + label_h)

    frames = []
    for f in range(n):
        sheet = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(sheet)
        for k, (g, im) in enumerate(zip(gifs, ims)):
            im.seek(f)
            fr = im.convert("RGB")
            # each source GIF quantised its background slightly differently: map it to one shared colour
            arr = np.array(fr)
            arr[(arr == arr[0, 0]).all(axis=-1)] = BG
            fr = Image.fromarray(arr)
            s = min(cell / fr.width, cell / fr.height)
            fr = fr.resize((max(1, round(fr.width * s)), max(1, round(fr.height * s))), Image.NEAREST)
            x0, y0 = (k % cols) * cell, (k // cols) * (cell + label_h)
            sheet.paste(fr, (x0 + (cell - fr.width) // 2, y0 + (cell - fr.height)))
            text = LABELS.get(g.parent.name, g.parent.name)
            tw = d.textlength(text, font=font)
            d.text((x0 + (cell - tw) / 2, y0 + cell + 6), text, fill=INK, font=font)
        frames.append(sheet)

    strip = Image.new("RGB", (W, H * 4))
    for i, f in enumerate(frames[:: max(1, n // 4)][:4]):
        strip.paste(f, (0, i * H))
    pal = strip.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE)
    q = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=duration, loop=0, disposal=1, optimize=False)
    print(f"{args.out} {len(q)} frames {W}x{H} {args.out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
