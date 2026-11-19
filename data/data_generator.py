"""Generate the synthetic hand-drawn shape dataset (square, circle, triangle; 70x70 grayscale PNG, train/validation/test).

Run: python data_generator.py   (writes data/shapes_dataset/; default seed 42)
"""

from pathlib import Path
import argparse
import csv
import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "shapes_dataset"

WIDTH = 70
HEIGHT = 70

IMAGES_PER_CLASS = 40000

TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

SHAPES = ["square", "circle", "triangle"]

MIN_SIZE = 0.50
MAX_SIZE = 0.85

MIN_THICKNESS = 1
MAX_THICKNESS = 4

MIN_ROTATION = 0
MAX_ROTATION = 360

POSITION_VARIATION = 0.25

SUPERSAMPLING = 4
RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# HAND-DRAWN LOOK
#
# NOISE_LEVEL is a master dial that multiplies every effect below:
#   0.0 = perfectly clean shapes (like the old script, minus roughness)
#   1.0 = default hand-drawn look
#   2.0+ = very messy
# Distances are in "shape units" (a circle has radius 1), so they are
# independent of image size.
# -----------------------------------------------------------------------------

NOISE_LEVEL = 1.0

# --- Geometry: how the outline itself is distorted ---
ROUGHEN_AMOUNT = 0.07        # random radial push on each original vertex
RESAMPLE_SPACING = 0.04      # outline is subdivided into points this far apart,
                             # so straight edges can wobble (smaller = finer)
WOBBLE_AMPLITUDE = 0.08      # smooth, wavy displacement along the outline
WOBBLE_KNOTS = 7             # wiggles per outline (low = slow waves)
JITTER = 0.018               # high-frequency shake, per point (shaky hand)

AFFINE_SCALE = 0.20          # random anisotropic stretch (+/- fraction)
AFFINE_SHEAR = 0.15          # random skew
WARP_AMPLITUDE = 0.08        # low-frequency bend of the whole shape
                             # (straight lines become curved, etc.)

# --- Stroke: how the "pen" behaves ---
OPEN_GAP_MAX = 0.06          # max fraction of the outline left unclosed
OVERSHOOT_MAX = 0.16         # max fraction drawn past the start point
END_DRIFT = 0.05             # end of the stroke misses the start by this much
SECOND_STROKE_PROBABILITY = 0.25   # chance of a sketchy second pass
SECOND_STROKE_OFFSET = 0.07  # how far the second pass strays from the first

THICKNESS_VARIATION = 0.50   # pen pressure: width varies along the line
THICKNESS_KNOTS = 9
INTENSITY_VARIATION = 60     # ink darkness variation (0-255 grey levels)

# --- Image: paper / scan artefacts ---
BLUR_MAX = 0.15
PIXEL_NOISE_STD = 2.0
PAPER_TEXTURE = 4.0
SPECKLE_COUNT = 2
SPECKLE_STRENGTH = 40


# =============================================================================
# SHAPE DEFINITIONS
# =============================================================================

def get_shape_points(shape):

    if shape == "square":
        return [(-1, -1), (1, -1), (1, 1), (-1, 1)]

    if shape == "rectangle":
        return [(-1, -0.6), (1, -0.6), (1, 0.6), (-1, 0.6)]

    if shape == "triangle":
        return [(0, -1), (0.866, 0.5), (-0.866, 0.5)]

    if shape == "circle":
        return [
            (math.cos(2 * math.pi * i / 120),
             math.sin(2 * math.pi * i / 120))
            for i in range(120)
        ]

    if shape == "ellipse":
        return [
            (math.cos(2 * math.pi * i / 120),
             0.6 * math.sin(2 * math.pi * i / 120))
            for i in range(120)
        ]

    if shape == "figure_eight":
        return [
            (math.sin(2 * 2 * math.pi * i / 240),
             math.cos(2 * math.pi * i / 240))
            for i in range(240)
        ]

    raise ValueError(f"Unknown shape: {shape}")


# =============================================================================
# NOISE HELPERS
# =============================================================================

def smooth_noise(n, np_rng, knots):
    """Smooth 1D noise of length n with roughly `knots` random control values
    (cosine interpolated, wraps around so closed outlines have no seam)."""
    knots = max(2, int(knots))
    values = np_rng.normal(size=knots)
    x = np.linspace(0, knots, n, endpoint=False)
    i0 = np.floor(x).astype(int)
    t = x - i0
    t = (1 - np.cos(np.pi * t)) / 2
    return values[i0 % knots] * (1 - t) + values[(i0 + 1) % knots] * t


def multi_octave_noise(n, np_rng, knots):
    """Slow big waves plus faster smaller ones: looks more organic."""
    return (smooth_noise(n, np_rng, knots)
            + 0.5 * smooth_noise(n, np_rng, knots * 2.7)
            + 0.25 * smooth_noise(n, np_rng, knots * 6))


def roughen_points(points, rng, amount):
    points = np.array(points, dtype=np.float64)
    centre = points.mean(axis=0)
    vectors = points - centre
    noise = np.array([rng.uniform(-amount, amount) for _ in points])
    return centre + vectors * (1 + noise[:, None])


def resample_closed(points, spacing):
    """Subdivide a closed polygon into evenly spaced points."""
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    n = max(int(total / spacing), 12)
    t = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t, cum, closed[:, 0])
    y = np.interp(t, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def outline_normals(points):
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(norm, 1e-9)
    return np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)


# =============================================================================
# HAND-DRAWN DISTORTIONS
# =============================================================================

def warp_shape(points, np_rng, level):
    """Distortion shared by every pen pass: stretch, skew and smooth bending."""
    sx = 1 + np_rng.uniform(-AFFINE_SCALE, AFFINE_SCALE) * level
    sy = 1 + np_rng.uniform(-AFFINE_SCALE, AFFINE_SCALE) * level
    shear = np_rng.uniform(-AFFINE_SHEAR, AFFINE_SHEAR) * level
    matrix = np.array([[sx, shear], [0.0, sy]])
    points = points @ matrix.T

    amp = WARP_AMPLITUDE * level
    f1, f2 = np_rng.uniform(1.0, 3.0, 2)
    p1, p2 = np_rng.uniform(0, 2 * math.pi, 2)
    x, y = points[:, 0].copy(), points[:, 1].copy()
    points[:, 0] = x + amp * np.sin(f1 * y + p1)
    points[:, 1] = y + amp * np.sin(f2 * x + p2)
    return points


def wobble_outline(points, np_rng, level):
    """Wavy + shaky displacement perpendicular to the outline."""
    n = len(points)
    normals = outline_normals(points)
    wave = multi_octave_noise(n, np_rng, WOBBLE_KNOTS) * WOBBLE_AMPLITUDE * level
    shake = np_rng.normal(0, JITTER * level, n)
    return points + normals * (wave + shake)[:, None]


def open_stroke(points, np_rng, level):
    """Turn the closed outline into a pen stroke that starts somewhere random,
    may not quite close, may overshoot, and drifts off target."""
    n = len(points)
    start = int(np_rng.integers(n))
    points = np.roll(points, -start, axis=0)

    fraction = np_rng.uniform(-OPEN_GAP_MAX, OVERSHOOT_MAX) * level
    fraction = max(fraction, -0.4)
    k = int(abs(fraction) * n)

    if fraction >= 0:
        points = np.vstack([points, points[:k + 1]])
    else:
        points = points[:max(n - k, 4)]

    drift = np_rng.normal(0, END_DRIFT * level, 2)
    ramp = np.linspace(0, 1, len(points))[:, None]
    return points + ramp * drift


def postprocess(image, np_rng, level):
    """Paper texture, blur, pixel noise and stray dots."""
    blur = np_rng.uniform(0, BLUR_MAX) * level
    if blur > 0.05:
        image = image.filter(ImageFilter.GaussianBlur(blur))

    arr = np.asarray(image, dtype=np.float32)

    if PAPER_TEXTURE > 0 and level > 0:
        small = np_rng.normal(0, PAPER_TEXTURE * level, (4, 5)).astype(np.float32)
        texture = Image.fromarray(small, mode="F").resize(
            (WIDTH, HEIGHT), Image.Resampling.BICUBIC
        )
        arr += np.asarray(texture)

    if PIXEL_NOISE_STD > 0 and level > 0:
        arr += np_rng.normal(0, PIXEL_NOISE_STD * level, arr.shape)

    n_specks = int(np_rng.poisson(SPECKLE_COUNT * level))
    for _ in range(n_specks):
        x = int(np_rng.integers(WIDTH))
        y = int(np_rng.integers(HEIGHT))
        arr[y, x] -= np_rng.uniform(0.3, 1.0) * SPECKLE_STRENGTH

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L"), round(float(blur), 2)


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def generate_image(shape, rng):

    ss = SUPERSAMPLING
    level = rng.triangular(0.25, 1.1, 0.65)

    # numpy generator derived from the main rng, so results stay reproducible
    np_rng = np.random.default_rng(rng.randrange(2 ** 32))

    # --- outline geometry (in shape units) ---
    points = np.array(get_shape_points(shape), dtype=np.float64)
    points = roughen_points(points, rng, ROUGHEN_AMOUNT * level)
    points = resample_closed(points, RESAMPLE_SPACING)
    points = warp_shape(points, np_rng, level)

    n_passes = 2 if rng.random() < SECOND_STROKE_PROBABILITY * min(level, 1.0) else 1

    paths = []
    for p in range(n_passes):
        path = points.copy()
        if p > 0:
            # a second, sloppier pass over roughly the same outline
            offset = smooth_noise(len(path), np_rng, 4) * SECOND_STROKE_OFFSET * level
            path = path + outline_normals(path) * offset[:, None]
        path = wobble_outline(path, np_rng, level)
        path = open_stroke(path, np_rng, level)
        paths.append(path)

    # --- rotate ---
    angle = math.radians(rng.uniform(MIN_ROTATION, MAX_ROTATION))
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]
    ])
    paths = [p @ rotation.T for p in paths]

    # --- scale to a random size, but keep everything inside the image ---
    stacked = np.vstack(paths)
    extent = np.ptp(stacked, axis=0).max()
    size = rng.uniform(MIN_SIZE, MAX_SIZE) * min(WIDTH, HEIGHT)
    scale = size / extent

    margin = MAX_THICKNESS + 2

    scale = min(
        scale,
        (WIDTH - 2 * margin) / np.ptp(stacked[:, 0]),
        (HEIGHT - 2 * margin) / np.ptp(stacked[:, 1])
    )

    # centre the bounding box on the origin, then scale
    bbox_centre = (stacked.max(axis=0) + stacked.min(axis=0)) / 2
    paths = [(p - bbox_centre) * scale for p in paths]
    stacked = np.vstack(paths)
    shape_w, shape_h = np.ptp(stacked, axis=0)

    # --- position ---
    cx = WIDTH / 2 + rng.uniform(
        -POSITION_VARIATION * WIDTH,
        POSITION_VARIATION * WIDTH
    )
    cy = HEIGHT / 2 + rng.uniform(
        -POSITION_VARIATION * HEIGHT,
        POSITION_VARIATION * HEIGHT
    )

    cx = float(np.clip(cx, margin + shape_w / 2, WIDTH - margin - shape_w / 2))
    cy = float(np.clip(cy, margin + shape_h / 2, HEIGHT - margin - shape_h / 2))

    paths = [(p + [cx, cy]) * ss for p in paths]

    # --- pen properties ---
    background = rng.randint(220, 255)
    foreground = rng.randint(0, 50)
    thickness = rng.randint(MIN_THICKNESS, MAX_THICKNESS)

    image = Image.new("L", (WIDTH * ss, HEIGHT * ss), background)
    draw = ImageDraw.Draw(image)

    for p, path in enumerate(paths):
        m = len(path)

        # pen pressure -> varying width
        width_factor = 1 + smooth_noise(m, np_rng, THICKNESS_KNOTS) * THICKNESS_VARIATION * level
        widths = thickness * ss * np.clip(width_factor, 0.3, 2.5)
        if p > 0:
            widths = widths * 0.7

        # ink darkness variation
        fade = np.abs(smooth_noise(m, np_rng, THICKNESS_KNOTS)) * INTENSITY_VARIATION * level
        shades = np.clip(foreground + fade, 0, background - 40).astype(int)
        if p > 0:
            shades = np.clip(shades + 25, 0, background - 30)

        for i in range(m - 1):
            a = tuple(path[i])
            b = tuple(path[i + 1])
            w = widths[i]
            fill = int(shades[i])
            draw.line([a, b], fill=fill, width=max(1, int(round(w))))
            r = w / 2
            draw.ellipse([a[0] - r, a[1] - r, a[0] + r, a[1] + r], fill=fill)

    image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    image, blur = postprocess(image, np_rng, level)

    return image, {
        "rotation": round(angle * 180 / math.pi, 2),
        "thickness": thickness,
        "size": round(size, 2),
        "background": background,
        "foreground": foreground,
        "center_x": round(cx, 2),
        "center_y": round(cy, 2),
        "passes": n_passes,
        "blur": blur,
        "noise_level": level
    }


# =============================================================================
# PREVIEW
# =============================================================================

def save_preview(preview, path, zoom=3):
    """Contact sheet: one row per class, so you can tune the noise quickly."""
    cols = max(len(v) for v in preview.values())
    rows = len(preview)
    pad = 4
    cell_w, cell_h = WIDTH * zoom, HEIGHT * zoom
    sheet = Image.new(
        "L",
        (cols * (cell_w + pad) + pad, rows * (cell_h + pad) + pad),
        128
    )
    for r, shape in enumerate(preview):
        for c, img in enumerate(preview[shape]):
            big = img.resize((cell_w, cell_h), Image.Resampling.NEAREST)
            sheet.paste(big, (pad + c * (cell_w + pad), pad + r * (cell_h + pad)))
    sheet.save(path)


# =============================================================================
# DATASET GENERATION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a hand-drawn shapes dataset.")
    parser.add_argument("--noise-level", type=float, default=None,
                        help="master noise dial (0 = clean, 1 = default, 2+ = messy)")
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():

    global NOISE_LEVEL, IMAGES_PER_CLASS, OUTPUT_DIR, RANDOM_SEED

    args = parse_args()
    if args.noise_level is not None:
        NOISE_LEVEL = args.noise_level
    if args.images_per_class is not None:
        IMAGES_PER_CLASS = args.images_per_class
    if args.output is not None:
        OUTPUT_DIR = args.output
    if args.seed is not None:
        RANDOM_SEED = args.seed

    rng = random.Random(RANDOM_SEED)
    output = Path(OUTPUT_DIR)

    if not math.isclose(TRAIN_RATIO + VAL_RATIO + TEST_RATIO, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    rows = []
    preview = {}

    for label, shape in enumerate(SHAPES):

        print(f"Generating {shape}...")

        samples = []

        for i in range(IMAGES_PER_CLASS):
            image, metadata = generate_image(shape, rng)
            samples.append((image, metadata, i))

        preview[shape] = [img for img, _, _ in samples[:8]]

        rng.shuffle(samples)

        n_train = int(IMAGES_PER_CLASS * TRAIN_RATIO)
        n_val = int(IMAGES_PER_CLASS * VAL_RATIO)

        splits = {
            "train": samples[:n_train],
            "validation": samples[n_train:n_train + n_val],
            "test": samples[n_train + n_val:]
        }

        for split, split_samples in splits.items():

            folder = output / split / shape
            folder.mkdir(parents=True, exist_ok=True)

            for image, metadata, index in split_samples:

                filename = f"{shape}_{index:05d}.png"
                image.save(folder / filename)

                rows.append({
                    "filename": filename,
                    "path": str((folder / filename).relative_to(output)),
                    "label": label,
                    "class": shape,
                    "split": split,
                    **metadata
                })

    with open(output / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(output / "class_mapping.txt", "w") as f:
        for label, shape in enumerate(SHAPES):
            f.write(f"{label}: {shape}\n")

    save_preview(preview, output / "preview.png")

    print(f"\nGenerated {len(rows)} images in {output.resolve()}")
    print(f"Noise level: {NOISE_LEVEL}  (see preview.png)")


if __name__ == "__main__":
    main()