#!/usr/bin/env python3
"""Export a trained checkpoint and repository validation images for HLS.

Writes nn_weights.h and nn_calibration.h to train_model/runs/export by default.
Example command from the repo root.
    python train_model/export_hls.py selector --checkpoint train_model/checkpoints/selector

For a head, the selector it was trained on is read from the 'selector' entry of its metrics.json
(written by train_head.py, relative to train_model/); --selector overrides it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from augment_data import CLASSES, image_files, load_images
from integer_reference import IntegerCNN, IntegerSpiking, check_snn_spec
from train_cnn import CNN
from train_common import RUNS
from train_head import Head, PAIRS
from train_snn import SNN


# ===================== EDIT THESE =====================
OUTPUT_ROOT = RUNS / "export"
LANES = 6
VALIDATION_VECTORS = 30
SNN_SYNTHETIC_VECTORS_PER_CLASS = 3
# Match the synthetic validation subset in train_common.Data.
SYNTHETIC_SUBSET_PER_CLASS = 3000
SYNTHETIC_SUBSET_SEED = 20261118
NOISE_SEED, SNN_NOISE_SEED = 5, 910


# ---------------------- Checkpoints ----------------------
def metadata(folder):
    path = folder / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def load_model(folder, target):
    """Load the checkpoint into the architecture used by the HLS target."""
    if target == "snn":
        model = SNN()
    elif target.startswith("head_"):
        model = Head(10, 16, 24)
    elif target == "selector":
        model = CNN(6, 10, 12, 16, 4, 4)
    else:
        model = CNN(8, 14, 20, 24, 5, 4)
    filename = "head.pt" if isinstance(model, Head) else "model.pt"
    state = torch.load(folder / filename, map_location="cpu", weights_only=True)
    # SNN checkpoints can wrap the parameters and architecture spec together.
    if target == "snn":
        if "state" in state:
            check_snn_spec(state["spec"])
            state = state["state"]
    elif metadata(folder).get("spec", model.spec) != model.spec:
        raise ValueError(f"{folder} must use the HLS architecture {model.spec}")
    model.load_state_dict(state)
    return model.eval().requires_grad_(False)


def head_target(folder, selector):
    """Read the head's excluded class and locate its selector checkpoint."""
    meta = metadata(folder)
    excluded = meta.get("cls")
    if type(excluded) is not int or excluded not in range(len(CLASSES)):
        raise ValueError("head metrics.json must give the excluded class in 'cls' (0, 1 or 2)")
    if selector is None and meta.get("selector"):
        selector = RUNS.parent / meta["selector"]   # written by train_head.py, relative to train_model/
    if selector is None or not (selector / "model.pt").is_file():
        raise ValueError("Selector checkpoint not found. Pass its directory with --selector.")
    return f"head_{CLASSES[excluded]}", excluded, selector


# ---------------------- Calibration images ----------------------
def calibration_samples(target, excluded):
    """Sample labelled validation images and append three unlabelled stress inputs."""
    hand, labels = image_files("hand", "validation")
    classes = [index for index in range(len(CLASSES)) if index != excluded]
    rows = [np.flatnonzero(labels == index) for index in classes]
    count = VALIDATION_VECTORS // len(classes)
    if any(len(row) < count for row in rows):
        raise ValueError(f"Need at least {count} hand validation images per selected class")
    if target == "snn":
        hand_rows = np.stack([row[:count] for row in rows], axis=1).flatten()
    else:
        hand_rows = np.concatenate([row[np.linspace(0, len(row) - 1, count).astype(int)] for row in rows])
    paths = [hand[index] for index in hand_rows]
    targets = labels[hand_rows]

    if target == "snn":
        synthetic, syn_labels = image_files("synthetic", "validation")
        rng = np.random.default_rng(SYNTHETIC_SUBSET_SEED)
        subset = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(syn_labels == index), SYNTHETIC_SUBSET_PER_CLASS, replace=False)
            for index in range(len(CLASSES))
        ]))
        syn_rows = np.concatenate([
            subset[syn_labels[subset] == index][:SNN_SYNTHETIC_VECTORS_PER_CLASS] for index in classes
        ])
        paths.extend(synthetic[index] for index in syn_rows)
        targets = np.r_[targets, syn_labels[syn_rows]]
        noise = torch.randint(256, (70, 70), dtype=torch.uint8,
                              generator=torch.Generator().manual_seed(SNN_NOISE_SEED)).numpy()
    else:
        noise = np.random.default_rng(NOISE_SEED).integers(0, 256, (70, 70), dtype=np.uint8)

    if excluded is not None:
        targets = (targets == PAIRS[excluded][0]).astype(int)
    extremes = np.stack([np.full_like(noise, 255), np.zeros_like(noise), noise])
    return np.concatenate([load_images(paths), extremes]), np.r_[targets, [-1] * len(extremes)]


# ---------------------- Header formatting ----------------------
def array_decl(name, values, ctype, dims=None, *, storage="static const", row_separator=","):
    """Format a numeric array as a C declaration with nested initializers."""
    values = np.asarray(values)
    if dims is None:
        dims = "".join(f"[{size}]" for size in values.shape)

    def initializer(array):
        if array.ndim == 1:
            return "{" + ",".join(map(str, array.tolist())) + "}"
        return "{" + row_separator.join(initializer(row) for row in array) + "}"

    return f"{storage} {ctype} {name}{dims} = {initializer(values)};\n"


def weights_header(reference, spiking):
    """Pack each layer's parameters into the six HLS ROM lanes."""
    weight_banks, bias_banks = [], []
    for weights, bias in reference.parameters.values():
        weights = weights.flatten(1).numpy()
        groups = (len(weights) + LANES - 1) // LANES
        padded = groups * LANES - len(weights)
        # Channel c goes in lane c % LANES, slot c // LANES, with zero-filled tails.
        bank = np.pad(weights, ((0, padded), (0, 0))).reshape(groups, LANES, -1)
        weight_banks.append(bank.transpose(1, 0, 2).reshape(LANES, -1))
        bias_banks.append(np.pad(bias.numpy(), (0, padded)).reshape(groups, LANES).T)
    weights = np.concatenate(weight_banks, axis=1)
    biases = np.concatenate(bias_banks, axis=1)
    if spiking:
        return ("// Generated weights and biases only.\n#pragma once\n"
                + array_decl("WEIGHT_ROM", weights, "weight_t", row_separator=",\n")
                + array_decl("BIAS_ROM", biases, "ap_int<32>", row_separator=",\n"))
    return ("// Generated by export_hls.py. Layout follows the NETWORK CONFIGURATION of nn.cpp.\n\n"
            + array_decl("WEIGHT_ROM", weights, "weight_t", storage="const")
            + "\n" + array_decl("BIAS_ROM", biases, "accumulator_t", storage="const"))


def run_length_encode(images):
    """Store equal pixel runs and the start offset of each image's runs."""
    values, lengths, starts = [], [], [0]
    for image in images:
        pixels = image.ravel()
        edges = np.r_[0, np.flatnonzero(pixels[1:] != pixels[:-1]) + 1, len(pixels)]
        values.extend(pixels[edges[:-1]].tolist())
        lengths.extend(np.diff(edges).tolist())
        starts.append(len(values))
    return values, lengths, starts


def calibration_header(images, labels, scores, features, target):
    """Write the testbench inputs and expected Q16.16 scores."""
    text = ["// Generated by export_hls.py. Expected integer Q16.16 scores.\n#pragma once\n#include <stdint.h>\n",
            f"#define CALIBRATION_COUNT {len(images)}\n"]
    spiking = target == "snn"
    if not target.startswith("head_"):
        values, lengths, starts = run_length_encode(images)
        if not spiking:
            text.append(f"#define CALIBRATION_RUN_COUNT {len(values)}\n")
        runs = None if spiking else "[CALIBRATION_RUN_COUNT]"
        start_dims = None if spiking else "[CALIBRATION_COUNT + 1]"
        text += [f"#define CALIBRATION_OUTPUTS {scores.shape[1]}\n",
                 array_decl("calibration_run_values", values, "uint8_t", runs),
                 array_decl("calibration_run_lengths", lengths, "uint16_t", runs),
                 array_decl("calibration_run_starts", starts, "uint32_t", start_dims)]
    else:
        text.append("#define CALIBRATION_OUTPUTS 1\n")
    if features is not None:
        flat = features.flatten(1).numpy()
        text += [f"#define CALIBRATION_FEATURE_COUNT {flat.shape[1]}\n",
                 array_decl("calibration_features", flat, "int16_t",
                            "[CALIBRATION_COUNT][CALIBRATION_FEATURE_COUNT]")]
    if spiking:
        text += [array_decl("calibration_labels", labels, "int8_t"),
                 array_decl("calibration_scores", scores.numpy(), "int32_t", row_separator=",\n")]
    else:
        text += [array_decl("calibration_scores", scores.numpy(), "int32_t",
                            "[CALIBRATION_COUNT][CALIBRATION_OUTPUTS]"),
                 array_decl("calibration_labels", labels, "int32_t", "[CALIBRATION_COUNT]")]
    return "".join(text)


# ---------------------- Main ----------------------
@torch.inference_mode()
def export(args):
    """Compute both headers before writing them to the output directory."""
    target, excluded = args.network, None
    if target == "head":
        target, excluded, selector = head_target(args.checkpoint, args.selector)
    out = args.out if args.out is not None else OUTPUT_ROOT / target
    paths = [out / name for name in ("nn_weights.h", "nn_calibration.h")]
    vitis = (RUNS.parent.parent / "Vitis").resolve()
    if any(path.resolve().is_relative_to(vitis) for path in [out, *paths]):
        raise ValueError("Choose an output directory outside Vitis/")

    model = load_model(args.checkpoint, target)
    images, labels = calibration_samples(target, excluded)
    pixels = torch.from_numpy(images[:, None])
    features = None
    if target == "snn":
        reference = IntegerSpiking(model)
        scores = reference.scores(pixels)
    else:
        reference = IntegerCNN(model, target)
        trunk = reference if excluded is None else IntegerCNN(load_model(selector, "selector"), "selector")
        maps = trunk.features(pixels)
        scores = reference.scores(maps)
        features = maps if target != "cnn" else None
        for name, count in reference.clipped.items():
            print(f"{name}: clipped {count} weights to the HLS precision")

    headers = (weights_header(reference, target == "snn"),
               calibration_header(images, labels, scores, features, target))
    out.mkdir(parents=True, exist_ok=True)
    for path, header in zip(paths, headers):
        path.write_text(header)
    print(f"Exported {target} to {out} ({len(images)} calibration vectors)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("network", choices=("cnn", "snn", "selector", "head"))
    parser.add_argument("--checkpoint", type=Path, required=True, help="directory containing model.pt or head.pt")
    parser.add_argument("--out", type=Path, help="output directory (default runs/export/<network>)")
    parser.add_argument("--selector", type=Path, help="head's selector directory, overriding metrics.json")
    args = parser.parse_args()
    if args.selector is not None and args.network != "head":
        parser.error("--selector is only used with head")
    try:
        export(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
