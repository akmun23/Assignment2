#!/usr/bin/env python3
"""Train the fully spiking 70x70 shape classifier with quantisation in the forward pass."""

import json

import torch
from torch import nn
import torch.nn.functional as F

from train_cnn import load_cnn
from train_common import (RUNS, TEACHER_DIR, Data, classification_loss, classifier_metrics,
                          predict, save_model, setup, train)


# ===================== EDIT THESE =====================
OUTPUT_DIR = RUNS / "snn"
TEACHER = TEACHER_DIR
SEED = 1
EPOCHS = 120
SAMPLES = 30_000
BATCH = 256
LR = 4e-3

STEPS = 6
FRACTION = 64  # Current and membrane are stored in units of 1/64.
LAYERS = (
    # kind, output channels, kernel, stride, padding
    ("conv", 12, 9, 6, 1),
    ("dw", 12, 3, 2, 1),
    ("conv", 6, 1, 1, 0),
    ("linear", 96, 1, 1, 0),
    ("linear", 3, 1, 1, 0),
)


# ---------------------- Model ----------------------
def round_units(value, scale, limits=None):
    discrete = torch.floor(value * scale + 0.5)
    if limits is not None:
        discrete = discrete.clamp(*limits)
    return discrete.detach() / scale + (value - value.detach())


class Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        return gradient / (1 + 2 * voltage.abs()).square()


def lif(currents, decay_shift):
    membrane = torch.zeros_like(currents[0])
    spikes = []
    for current in currents:
        leaked = membrane / (1 << decay_shift)
        membrane = (torch.floor(leaked.detach() * FRACTION) / FRACTION
                    + (leaked - leaked.detach())) + current
        fired = Spike.apply(membrane - 1)
        membrane = membrane - fired.detach()
        spikes.append(fired)
    return torch.stack(spikes), membrane


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.spec = dict(name="w_p6h96", steps=STEPS, readout="count_residual", slope=2.0, layers=[])
        width, inputs = 70, 1
        for index, (kind, outputs, kernel, stride, padding) in enumerate(LAYERS):
            if kind == "linear":
                layer = nn.Linear(width**2 * inputs, outputs)
                width = 1
            else:
                groups = inputs if kind == "dw" else 1
                layer = nn.Conv2d(inputs, outputs, kernel, stride=stride, padding=padding, groups=groups)
                width = (width + 2 * padding - kernel) // stride + 1
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.constant_(layer.bias, 0.25)
            if kind == "dw":
                with torch.no_grad():
                    layer.weight.mul_(0.1)
                    layer.weight[:, 0, kernel // 2, kernel // 2] += 1
            inputs = outputs
            self.layers.append(layer)
            self.spec["layers"].append(dict(kind=kind, outputs=outputs, kernel=kernel,
                                            stride=stride, padding=padding, pool=1, threshold_exp=6,
                                            decay_shift=int(index < len(LAYERS) - 1)))

    def current(self, index, inputs):
        layer = self.layers[index]
        weights = round_units(layer.weight, FRACTION, (-128, 127))
        bias_scale, bias_limit = (FRACTION * 256, 1 << 20) if index == 0 else (FRACTION, 1 << 15)
        bias = round_units(layer.bias, bias_scale, (-bias_limit, bias_limit - 1))
        if isinstance(layer, nn.Linear):
            current = F.linear(inputs.flatten(1), weights, bias)
        else:
            current = F.conv2d(inputs, weights, bias, layer.stride, layer.padding, groups=layer.groups)
        return round_units(current, FRACTION)

    def forward(self, images):
        # The stem converts the raw image to a constant current for all six steps.
        ink = round_units((1 - images) * (255 / 256), 256)
        current = self.current(0, ink)
        spikes, membrane = lif(current.unsqueeze(0).expand(STEPS, *current.shape), decay_shift=1)
        for index, config in enumerate(self.spec["layers"][1:], start=1):
            current = self.current(index, spikes.flatten(0, 1))
            current = current.reshape(STEPS, len(images), *current.shape[1:])
            spikes, membrane = lif(current, config["decay_shift"])

        # Read out spike count plus residual membrane, floored to Q16.16 as in HLS.
        count = spikes.sum(dim=0)
        value = (count + membrane) * (4 / STEPS)
        units = count.double() * FRACTION + (membrane.double() * FRACTION).round()
        discrete = torch.floor(units * 262144 / (STEPS * FRACTION)) / 65536
        return discrete.detach() + (value.double() - value.detach().double())


# ---------------------- Main ----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    device = setup()
    torch.manual_seed(SEED)
    # Full float32 products preserve the quantised forward pass on CUDA.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = SNN().to(device)
    teacher = load_cnn(TEACHER, device)
    data = Data(device)

    def loss(logits, images, labels):
        return classification_loss(logits, labels, teacher(images))

    model, history, epoch = train(model, data, loss, classifier_metrics, epochs=EPOCHS,
                                  samples=SAMPLES, batch_size=BATCH, lr=LR, seed=SEED,
                                  validation_every=1, select_best=True)
    selected = history[epoch - 1]
    checkpoint = dict(spec=model.spec, state=model.state_dict(), epoch=epoch, validation=selected)
    logits = {name: predict(model, images, BATCH) for name, (images, _) in data.validation.items()}
    save_model(OUTPUT_DIR, checkpoint, history, data, role="snn", spec=model.spec,
               selected=selected, logits=logits)
    (OUTPUT_DIR / "spec.json").write_text(json.dumps(model.spec, indent=2) + "\n")
    print(f"Saved SNN from epoch {epoch} to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
