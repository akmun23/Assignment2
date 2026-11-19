#!/usr/bin/env python3
"""Train pair heads on the feature map of a saved selector."""

import os

import torch
from torch import nn
import torch.nn.functional as F

from augment_data import CLASSES
from train_cnn import load_cnn
from train_common import CHECKPOINTS, RUNS, Data, predict, save_model, setup, train


# ===================== EDIT THESE =====================
SELECTOR = CHECKPOINTS / "selector"
OUTPUT_DIR = RUNS / "heads"
EXCLUDED_CLASSES = CLASSES
EPOCHS = 100
SAMPLES = 30_000
BATCH = 256
LR = 4e-3

# Each head is named after the omitted shape. A nonnegative score picks the first kept class.
PAIRS = ((1, 2), (0, 2), (0, 1))


# ---------------------- Model ----------------------
class Head(nn.Module):
    def __init__(self, inputs=10, channels=16, hidden=24):
        super().__init__()
        self.spec = f"head_b{inputs}c{channels}h{hidden}"
        self.dw2 = nn.Conv2d(inputs, inputs, 3, padding=1, groups=inputs, bias=False)
        self.pw2 = nn.Conv2d(inputs, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(4 * 4 * channels, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, features):
        x = F.relu(self.bn2(self.pw2(self.dw2(features))))
        x = F.avg_pool2d(x, 2).flatten(1)
        return self.fc2(F.relu(self.fc1(x))).squeeze(1)


class FrozenSelectorHead(nn.Module):
    def __init__(self, selector, head):
        super().__init__()
        self.selector = selector
        self.head = head

    def train(self, mode=True):
        super().train(mode)
        self.selector.eval()  # Keep the selector's BatchNorm statistics fixed.
        return self

    def forward(self, images):
        return self.head(self.selector.features(images))


# ---------------------- Loss and validation ----------------------
def pair_loss(logits, labels, excluded, teacher_probs=None):
    first, second = PAIRS[excluded]
    keep = labels != excluded
    target = (labels[keep] == first).float() * 0.97 + 0.015
    if teacher_probs is not None:
        pair_total = teacher_probs[keep, first] + teacher_probs[keep, second]
        target = 0.5 * (target + teacher_probs[keep, first] / pair_total.clamp_min(1e-6))
    loss = F.binary_cross_entropy_with_logits(logits[keep], target, reduction="sum")
    return loss / keep.sum().clamp_min(1)


def pair_metrics(logits, labels, excluded):
    first, _ = PAIRS[excluded]
    keep = labels != excluded
    correct = (logits[keep] >= 0) == (labels[keep] == first)
    return dict(n=len(correct), acc=correct.float().mean().item())


def save_head(folder, head, excluded, selector_dir, data, logits, history):
    final = {name: pair_metrics(logits[name], labels, excluded)
             for name, (_, labels) in data.validation.items()}
    save_model(folder, head.state_dict(), history, data, role="pair", cls=excluded,
               spec=head.spec, selector=os.path.relpath(selector_dir, CHECKPOINTS.parent), final=final,
               logits=logits)


# ---------------------- Main ----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    device = setup()
    selector = load_cnn(SELECTOR, device)

    for shape in EXCLUDED_CLASSES:
        excluded = CLASSES.index(shape)
        seed = 17 * (excluded + 1)
        torch.manual_seed(seed)
        data = Data(device, excluded=excluded)
        head = Head(inputs=selector.pw1.out_channels).to(device)
        model = FrozenSelectorHead(selector, head)

        def loss(logits, _images, labels):
            return pair_loss(logits, labels, excluded)

        def metrics(logits, labels):
            return pair_metrics(logits, labels, excluded)

        print(f"Training head with {shape} excluded", flush=True)
        model, history, _ = train(model, data, loss, metrics, epochs=EPOCHS,
                                  samples=SAMPLES, batch_size=BATCH, lr=LR, seed=seed)
        logits = {name: predict(model, images) for name, (images, _) in data.validation.items()}
        save_head(OUTPUT_DIR / f"head_{shape}", model.head, excluded, SELECTOR, data, logits, history)
    print(f"Saved heads to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
