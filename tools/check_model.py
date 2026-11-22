"""Sanity checks for a sprite-stack model and its rendered turntable GIF.

  python check_model.py model.npz turntable.gif [--scale 4] [--max-colours 64]

Checks (exit 0 only if all pass):
  1. solid: the filled voxels form one 26-connected piece (else prints the piece count and sizes)
  2. islands: in 8 frames (every 45 deg), detached model pixel islands smaller than 4 art px (8-connected,
     ground and ground-shadow colours excluded)
  3. colours: distinct colours over all GIF frames <= --max-colours
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).parent))
from stack_render import BG, shadow_colours  # noqa: E402


def components_3d(occ: np.ndarray) -> np.ndarray:
    """Sizes of 26-connected components (union-find with pointer jumping, vectorised)."""
    flat = occ.ravel()
    idx = np.flatnonzero(flat)
    if len(idx) == 0:
        return np.array([], int)
    nz, ny, nx = occ.shape
    pos = -np.ones(flat.size, np.int64)
    pos[idx] = np.arange(len(idx))
    z, y, x = np.unravel_index(idx, occ.shape)
    a_list, b_list = [], []
    for dz in (0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue  # 13 forward offsets
                zz, yy, xx = z + dz, y + dy, x + dx
                ok = (zz < nz) & (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
                j = np.where(ok, (zz * ny + np.clip(yy, 0, ny - 1)) * nx + np.clip(xx, 0, nx - 1), 0)
                ok &= flat[j]
                a_list.append(np.arange(len(idx))[ok])
                b_list.append(pos[j[ok]])
    a = np.concatenate(a_list)
    b = np.concatenate(b_list)
    lab = np.arange(len(idx))
    while True:
        m = np.minimum(lab[a], lab[b])
        new = lab.copy()
        np.minimum.at(new, a, m)
        np.minimum.at(new, b, m)
        new = new[new]
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    return np.sort(np.bincount(np.unique(lab, return_inverse=True)[1]))[::-1]


def islands_2d(mask: np.ndarray) -> list:
    h, w = mask.shape
    seen = np.zeros_like(mask)
    sizes = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        st, n = [(y0, x0)], 0
        seen[y0, x0] = True
        while st:
            y, x = st.pop()
            n += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        st.append((yy, xx))
        sizes.append(n)
    return sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("gif", type=Path)
    ap.add_argument("--scale", type=int, default=4, help="integer upscale of the GIF")
    ap.add_argument("--max-colours", type=int, default=64)
    ap.add_argument("--min-island", type=int, default=4)
    a = ap.parse_args()
    ok_all = True

    occ = np.load(a.model)["mat"] > 0
    sizes = components_3d(occ)
    ok = len(sizes) == 1
    ok_all &= ok
    extra = "" if ok else f", piece sizes {sizes[:10].tolist()}"
    print(f"[{'PASS' if ok else 'FAIL'}] solid: {len(sizes)} piece(s) (26-connectivity), {int(occ.sum())} voxels{extra}")

    im = Image.open(a.gif)
    frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    nf = len(frames)
    ground = np.array([BG] + shadow_colours())
    picks = [k * nf // 8 for k in range(8)]
    counts = []
    for i in picks:
        f = frames[i][::a.scale, ::a.scale]
        is_ground = (f[..., None, :] == ground[None, None]).all(-1).any(-1)
        isl = islands_2d(~is_ground)
        counts.append(sum(1 for s in isl if s < a.min_island))
    ok = sum(counts) == 0
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] islands < {a.min_island} px at frames {picks}: {counts}")

    allc = np.unique(np.concatenate([f.reshape(-1, 3) for f in frames]), axis=0)
    ok = len(allc) <= a.max_colours
    ok_all &= ok
    print(f"[{'PASS' if ok else 'FAIL'}] colours: {len(allc)} distinct over {nf} frames (max {a.max_colours})")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
