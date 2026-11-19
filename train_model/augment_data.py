#!/usr/bin/env python3
"""Augment the training images and save a few examples for inspection."""

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ===================== EDIT THESE =====================
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "runs/augmentation"
CLASSES = ("circle", "square", "triangle")
IMAGES_PER_CLASS = 8
SEED = 42

# The handwritten split is by contributor, which is the second field in each filename.
HAND_CONTRIBUTORS = {
    "train": {"drt", "if5", "if8", "im1", "lrt", "lt1", "mrt", "nae", "nld",
              "u02", "u13", "u17", "u18", "u19", "vly"},
    "validation": {"frt", "if1", "if3", "u01", "u03", "u04", "u05", "u06", "u07",
                   "u08", "u09", "u10"},
}


# ---------------------- Data ----------------------
def contributor(path):
    return path.stem.split(".")[1]


def image_files(corpus, split):
    root = DATA_ROOT / ("shapes" if corpus == "hand" else Path("shapes_dataset") / split)
    paths, labels = [], []
    for label, shape in enumerate(CLASSES):
        folder = root / shape
        files = sorted(folder.glob("*.png"))
        if corpus == "hand":
            files = [path for path in files if contributor(path) in HAND_CONTRIBUTORS[split]]
        if not files:
            raise FileNotFoundError(f"No {split} images in {folder}")
        paths.extend(files)
        labels.extend([label] * len(files))
    return paths, np.asarray(labels, dtype=np.int64)


def load_images(paths):
    images = np.empty((len(paths), 70, 70), dtype=np.uint8)
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (70, 70):
            raise ValueError(f"Expected a 70x70 image at {path}")
        images[index] = image
    return images


# ---------------------- Augmentation ----------------------
def augment(images):
    """Rotate, flip, scale and shift grayscale images, then vary the stroke and noise."""
    count = len(images)
    device = images.device
    angle = torch.rand(count, device=device) * 2 * math.pi
    scale = torch.empty(count, device=device).uniform_(0.85, 1.12)
    flip = (torch.rand(count, device=device) < 0.5).float() * 2 - 1
    shift = torch.empty(count, 2, device=device).uniform_(-0.10, 0.10)
    cos, sin = angle.cos() / scale, angle.sin() / scale
    transform = torch.stack([
        torch.stack([cos * flip, -sin, shift[:, 0]], dim=1),
        torch.stack([sin * flip, cos, shift[:, 1]], dim=1),
    ], dim=1)
    keep = torch.rand(count, device=device) >= 0.85
    identity = images.new_tensor([[1, 0, 0], [0, 1, 0]])
    transform = torch.where(keep[:, None, None], identity, transform)
    grid = F.affine_grid(transform, images.shape, align_corners=False)

    # Shift white to zero so grid sampling gives a white border.
    warped = F.grid_sample(images - 1, grid, align_corners=False) + 1
    ink = 1 - warped
    stroke = torch.rand(count, 1, 1, 1, device=device)
    thick = F.max_pool2d(ink, 3, stride=1, padding=1)
    thin = -F.max_pool2d(-ink, 3, stride=1, padding=1)
    ink = torch.where(stroke < 0.15, thick, torch.where(stroke < 0.25, thin, ink))
    noise = torch.randn_like(ink) * (torch.rand(count, 1, 1, 1, device=device) * 0.03)
    return (1 - ink + noise).clamp_(0, 1)


# ---------------------- Main ----------------------
def main():
    torch.manual_seed(SEED)
    paths, labels = image_files("hand", "train")

    for label, shape in enumerate(CLASSES):
        folder = OUTPUT_DIR / shape
        folder.mkdir(parents=True, exist_ok=True)
        indices = np.flatnonzero(labels == label)[:IMAGES_PER_CLASS]
        originals = load_images([paths[index] for index in indices])
        copies = augment(torch.from_numpy(originals).unsqueeze(1).float() / 255)
        copies = (copies[:, 0] * 255).round().byte().numpy()
        for index, (original, copy) in enumerate(zip(originals, copies), start=1):
            for name, image in (("original", original), ("augmented", copy)):
                path = folder / f"{index:02d}_{name}.png"
                if not cv2.imwrite(str(path), image):
                    raise OSError(f"Could not save {path}")
        print(f"{shape}: {len(originals)} examples in {folder}")


if __name__ == "__main__":
    main()
