# Pixel art Indonesia

Eight Indonesian landmarks as rotating pixel art, made with sprite stacking: each model is a stack of flat top-down slices, drawn one above the other and turned together, which reads as 3D.

![Eight landmarks turning](all-landmarks.gif)

| Model | Place | Slices |
|---|---|---|
| Borobudur | Magelang, Central Java | 56 |
| Gedung Sate | Bandung | 64 |
| Lawang Sewu | Semarang | 60 |
| Monas | Jakarta | 126 |
| Prambanan | Yogyakarta | 70 |
| Panggung Sangga Buwana | Solo | 106 |
| Tugu Yogyakarta | Yogyakarta | 112 |
| Wisma 46 | Jakarta | 114 |

## What is in each folder

`models/<name>/` holds:

- `turntable.gif`: the model turning, 48 frames, rendered with `tools/stack_render.py`.
- `slices.png`: the model itself, one strip of slices with the ground slice on the left.
- `build.py`: builds the voxel grid in NumPy and writes `slices.png`. Six models have this.
- `views.py` and `views.png` (Monas and Wisma 46 only): these two were drawn as four flat views and carved into slices with the slicer from [danielwh2/sprite-stacking](https://github.com/danielwh2/sprite-stacking). That slicer is not included here.
- `render_gif.py`: the first, simpler renderer written for that model. Kept for reference; running it overwrites `turntable.gif` with the older look.

`tools/` holds:

- `stack_render.py`: the renderer. It draws each slice as a slab with thickness, so thin parts stay solid when the camera is low, and it adds a dark line where one part sits in front of another. `--pitch 25` sets the camera angle in degrees.
- `render_all.py`: renders all eight models with `stack_render.py`.
- `make_grid.py`: tiles every `turntable.gif` into `all-landmarks.gif`.
- `check_model.py` and `compare.py`: checks on a rendered model, and side-by-side sheets of several GIFs.

## Run it

Python 3 with Pillow and NumPy.

```bash
pip install pillow numpy
```

```bash
python models/borobudur/build.py
```

```bash
python tools/render_all.py
```

```bash
python tools/make_grid.py
```

`build.py` rebuilds one model's `slices.png`, `render_all.py` renders every model's GIF, and `make_grid.py` rebuilds the grid. Options after `render_all.py` are passed to the renderer, so `python tools/render_all.py --pitch 25` renders all eight from a lower camera.

## How they were made

An AI coding agent (Claude) wrote each `build.py` from a text brief and reference photos, rendered the model, looked at the frames and fixed what read wrongly, a few rounds per model. No image generator drew the slices: they are cross-sections of a voxel grid, so they agree with each other.

The models are simplified and not to scale.
