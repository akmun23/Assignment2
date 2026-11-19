#!/usr/bin/env python3
"""Train the standalone 70x70 shape CNN."""

import json
import re

import torch
from torch import nn
import torch.nn.functional as F

from train_common import (RUNS, TEACHER_DIR, Data, classification_loss, classifier_metrics,
                          predict, save_model, setup, train)


# ===================== EDIT THESE =====================
OUTPUT_DIR = RUNS / "cnn"
TEACHER = TEACHER_DIR
SEED = 1
EPOCHS = 120
SAMPLES = 30_000
BATCH = 256
LR = 4e-3


# ---------------------- Model ----------------------
class CNN(nn.Module):
    def __init__(self, stem=8, first=14, second=20, hidden=24, kernel=5, stride=4):
        super().__init__()
        self.spec = f"a{stem}b{first}c{second}h{hidden}k{kernel}s{stride}o3"
        self.stride = stride
        padding = (kernel - 3) // 2 if stride == 4 else (kernel - 1) // 2
        self.stem = nn.Conv2d(1, stem, kernel, stride=stride, padding=padding, bias=False)
        self.bn0 = nn.BatchNorm2d(stem)
        self.dw1 = nn.Conv2d(stem, stem, 3, padding=1, groups=stem, bias=False)
        self.pw1 = nn.Conv2d(stem, first, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(first)
        self.dw2 = nn.Conv2d(first, first, 3, padding=1, groups=first, bias=False)
        self.pw2 = nn.Conv2d(first, second, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(second)
        self.fc1 = nn.Linear(4 * 4 * second, hidden)
        self.fc2 = nn.Linear(hidden, 3)

    def features(self, images):
        # Invert white-background grayscale and build the shared 8x8 feature map.
        x = F.relu(self.bn0(self.stem(1 - images)))
        if self.stride == 2:
            x = F.max_pool2d(x, 2)
        x = F.relu(self.bn1(self.pw1(self.dw1(x))))
        return F.max_pool2d(x, 2)

    def classify(self, features):
        x = F.relu(self.bn2(self.pw2(self.dw2(features))))
        x = F.avg_pool2d(x, 2).flatten(1)
        return self.fc2(F.relu(self.fc1(x)))

    def forward(self, images):
        return self.classify(self.features(images))


def load_cnn(folder, device):
    spec = json.loads((folder / "metrics.json").read_text())["spec"]
    architecture = re.fullmatch(r"a(\d+)b(\d+)c(\d+)h(\d+)k(\d+)(?:s([24]))?o3", spec)
    if architecture is None:
        raise ValueError(f"Unsupported CNN checkpoint architecture {spec}")
    stem, first, second, hidden, kernel, stride = architecture.groups()
    model = CNN(*(int(value) for value in (stem, first, second, hidden, kernel, stride or 2)))
    model.load_state_dict(torch.load(folder / "model.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval().requires_grad_(False)


# ---------------------- Main ----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    device = setup()
    teacher = load_cnn(TEACHER, device)
    torch.manual_seed(SEED)
    model = CNN().to(device)
    data = Data(device)

    def loss(logits, images, labels):
        return classification_loss(logits, labels, teacher(images))

    model, history, _ = train(model, data, loss, classifier_metrics, epochs=EPOCHS,
                              samples=SAMPLES, batch_size=BATCH, lr=LR, seed=SEED)
    logits = {name: predict(model, images) for name, (images, _) in data.validation.items()}
    save_model(OUTPUT_DIR, model.state_dict(), history, data, role="cnn", cls=-1,
               spec=model.spec, final=history[-1], logits=logits)
    print(f"Saved CNN to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
