"""Render every models/<name>/slices.png into models/<name>/turntable.gif with tools/stack_render.py."""
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RENDERER = ROOT / "tools" / "stack_render.py"

# model folder: (number of slices in the strip, integer upscale)
MODELS = {
    "borobudur": (56, 4),
    "gedung-sate": (64, 4),
    "lawang-sewu": (60, 4),
    "monas": (126, 4),
    "prambanan": (70, 3),
    "sangga-buwana-solo": (106, 4),
    "tugu-yogyakarta": (112, 4),
    "wisma46-bni": (114, 4),
}


def main():
    extra = sys.argv[1:]  # passed through, for example: --pitch 25
    for name, (count, scale) in MODELS.items():
        folder = ROOT / "models" / name
        strip = folder / "slices.png"
        width = Image.open(strip).size[0]
        assert width % count == 0, f"{name}: strip width {width} is not a multiple of {count}"
        cmd = [sys.executable, str(RENDERER), "--strip", str(strip), "--count", str(count),
               "--out", str(folder / "turntable.gif"), "--scale", str(scale), *extra]
        subprocess.run(cmd, check=True, capture_output=True)
        gif = Image.open(folder / "turntable.gif")
        print(f"{name:20s} {gif.size[0]}x{gif.size[1]} {gif.n_frames} frames")


if __name__ == "__main__":
    main()
