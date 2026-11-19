"""Turn trained model parameters and grayscale tensors into HLS ROMs and scores.

Run train_model/export_hls.py to load a checkpoint and write nn_weights.h and
nn_calibration.h. This module supplies the quantised tensors and Q16.16 scores.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ===================== EDIT THESE =====================
# Layer order is stem, dw1, pw1, dw2, pw2, fc1, fc2. Heads start at dw2.
# Fractions preserve the HLS scales after folding 1/255 and 1/4 into the weights.
# Each entry gives accumulator bits, weight bits, weight fractions, activation bits, shifts.
CNN_FORMATS = {
    "cnn": (22, (8, 7, 8, 6, 8, 7, 7), (13, 6, 5, 5, 7, 7, 6),
            (8, 8, 10, 12, 7, 9), (9, 6, 3, 5, 10, 5)),
    "selector": (25, (8, 7, 8, 7, 7, 8, 7), (14, 6, 6, 6, 5, 7, 5),
                 (7, 10, 9, 11, 11, 10), (10, 4, 7, 6, 4, 9)),
    "head_circle": (25, (8,) * 4, (7, 6, 8, 7), (10,) * 3, (8, 4, 8)),
    "head_square": (25, (8,) * 4, (7, 7, 7, 7), (10,) * 3, (8, 6, 7)),
    "head_triangle": (26, (8,) * 4, (7, 6, 8, 7), (10,) * 3, (7, 5, 9)),
}
SELECTOR_FRACTION, SELECTOR_MAX = 5, 511  # Shared 8x8 feature map format.

SNN_STEPS = 6
SNN_FRACTION = 64  # Also the LIF threshold.
SNN_STEM_SHIFT = 8
SNN_ANALOG_BITS, SNN_CURRENT_BITS, SNN_MEMBRANE_BITS = 20, 13, 14
SNN_GEOMETRY = (
    # Kind, outputs, kernel, stride, padding.
    ("conv", 12, 9, 6, 1),
    ("dw", 12, 3, 2, 1),
    ("conv", 6, 1, 1, 0),
    ("linear", 96, 1, 1, 0),
    ("linear", 3, 1, 1, 0),
)
SNN_DECAYS = (1, 1, 1, 1, 0)


# ---------------------- Shared arithmetic ----------------------
def round_shift(values, shift):
    """Divide by a power of two and round to nearest, with ties upward."""
    return (values + (1 << (shift - 1))) >> shift


def apply_layer(inputs, layer, weights, bias):
    """Apply the layer geometry with the supplied integer parameters."""
    if isinstance(layer, nn.Linear):
        return F.linear(inputs.flatten(1), weights, bias)
    return F.conv2d(inputs, weights, bias, layer.stride, layer.padding, groups=layer.groups)


def accumulator_bounds(weights, bias, low, high):
    """Bound every partial sum for an input range that contains zero."""
    weights = weights.flatten(1)
    positive = weights.clamp_min(0).sum(1)
    negative = weights.clamp_max(0).sum(1)
    return bias + low * positive + high * negative, bias + high * positive + low * negative


def check_signed(low, high, bits, description):
    """Reject bounds outside the signed range of the corresponding HLS type."""
    limit = 1 << (bits - 1)
    if not ((low >= -limit).all() and (high < limit).all()):
        raise ValueError(f"{description} exceeds the HLS signed {bits}-bit range")


# ---------------------- CNN, selector and heads ----------------------
def folded_parameters(model, name):
    """Fold BatchNorm, pixel normalisation and average pooling into the weights and biases."""
    layer = getattr(model, name)
    weights, bias = layer.weight, layer.bias
    bn_name = {"stem": "bn0", "pw1": "bn1", "pw2": "bn2"}.get(name)
    if bn_name:
        bn = getattr(model, bn_name)
        gain = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weights = weights * gain[:, None, None, None]
        bias = bn.bias - bn.running_mean * gain
    if name == "stem":
        weights = weights / 255  # Input is integer ink.
    elif name == "fc1":
        weights = weights / 4  # Pooling produces sums.
    return weights, bias if bias is not None else weights.new_zeros(len(weights))


class IntegerCNN:
    """Run the CNN, selector or head with the target's fixed-point format."""

    @torch.inference_mode()
    def __init__(self, model, target):
        """Quantise each folded layer and check its accumulator bounds."""
        self.model = model
        self.parameters = {}
        self.requantisation = {}
        self.clipped = {}
        accumulator_bits, weight_bits, fractions, activation_bits, shifts = CNN_FORMATS[target]
        names = ("stem", "dw1", "pw1", "dw2", "pw2", "fc1", "fc2")
        low, high, fraction = 0, 255, 0
        if target.startswith("head_"):
            names = names[3:]
            high, fraction = SELECTOR_MAX, SELECTOR_FRACTION
        for index, (name, bits, weight_fraction) in enumerate(zip(names, weight_bits, fractions)):
            weights, bias = folded_parameters(model, name)
            fraction += weight_fraction
            limit = (1 << (bits - 1)) - 1
            weights = (weights * (2 ** weight_fraction)).round()
            bias = (bias * (2 ** fraction)).round()
            if not torch.isfinite(weights).all():
                raise ValueError(f"{name} has nonfinite folded weights")
            clipped = int((weights.abs() > limit).sum())
            if clipped:
                self.clipped[name] = clipped
            check_signed(bias, bias, accumulator_bits, f"{name} bias")
            weights, bias = weights.clamp(-limit, limit).long(), bias.long()
            self.parameters[name] = weights, bias
            lower, upper = accumulator_bounds(weights, bias, low, high)
            check_signed(lower, upper, accumulator_bits, f"{name} accumulator")
            if name == "fc2":
                self.score_shift = 16 - fraction
                check_signed(lower << self.score_shift, upper << self.score_shift, 32, "Q16.16 score")
                break
            shift, bits = shifts[index], activation_bits[index]
            fraction -= shift
            low, high = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if name.startswith("dw") else (0, (1 << bits) - 1)
            self.requantisation[name] = shift, low, high
            if name == "pw2":
                high *= 4

    def layer(self, inputs, name):
        """Form a layer sum and rescale it to its activation or score format."""
        values = apply_layer(inputs, getattr(self.model, name), *self.parameters[name])
        if name == "fc2":
            return values << self.score_shift
        shift, low, high = self.requantisation[name]
        return round_shift(values, shift).clamp(low, high)

    def features(self, images):
        """Convert grayscale pixels into the shared 8x8 integer feature map."""
        values = self.layer(255 - images.long(), "stem")
        values = self.layer(self.layer(values, "dw1"), "pw1")
        return F.max_pool2d(values, 2)

    def scores(self, features):
        """Classify the shared feature map and return Q16.16 scores."""
        values = self.layer(self.layer(features, "dw2"), "pw2")
        values = F.avg_pool2d(values, 2, divisor_override=1)
        return self.layer(self.layer(values, "fc1"), "fc2")


# ---------------------- Spiking network ----------------------
def check_snn_spec(spec):
    """Validate the HLS geometry for checkpoints with a tuple or list of SNN layers."""
    keys = ("kind", "outputs", "kernel", "stride", "padding", "pool", "threshold_exp", "decay_shift")
    geometry = tuple(tuple(layer[key] for key in keys) for layer in spec["layers"])
    expected = tuple(layer + (1, 6, decay) for layer, decay in zip(SNN_GEOMETRY, SNN_DECAYS))
    if spec["steps"] != SNN_STEPS or spec["readout"] != "count_residual" or geometry != expected:
        raise ValueError("SNN checkpoint does not match the six-step HLS architecture")


class IntegerSpiking:
    """Run the six-step spiking network with integer currents and membranes."""

    @torch.inference_mode()
    def __init__(self, model):
        """Quantise the SNN parameters and check current and membrane bounds."""
        self.model = model
        self.parameters = {}
        check_snn_spec(model.spec)
        for index, (layer, decay) in enumerate(zip(model.layers, SNN_DECAYS)):
            if not (torch.isfinite(layer.weight).all() and torch.isfinite(layer.bias).all()):
                raise ValueError(f"SNN layer {index} has nonfinite parameters")
            stem = index == 0
            bias_scale = SNN_FRACTION << SNN_STEM_SHIFT if stem else SNN_FRACTION
            bias_limit = 1 << (20 if stem else 15)
            weights = (layer.weight.double() * SNN_FRACTION + 0.5).floor().clamp(-128, 127).long()
            bias = (layer.bias.double() * bias_scale + 0.5).floor().clamp(-bias_limit, bias_limit - 1).long()
            low, high = accumulator_bounds(weights, bias, 0, 255 if stem else 1)
            if stem:
                check_signed(low, high + (1 << (SNN_STEM_SHIFT - 1)), SNN_ANALOG_BITS, "SNN stem sum")
                low, high = round_shift(low, SNN_STEM_SHIFT), round_shift(high, SNN_STEM_SHIFT)
            check_signed(low, high, SNN_CURRENT_BITS, f"SNN layer {index} current")
            self.check_membrane(low, high, decay, index)
            self.parameters[index] = weights, bias

    @staticmethod
    def check_membrane(current_low, current_high, decay, index):
        """Check membrane bounds before the subtractive reset at each time step."""
        low = high = torch.zeros_like(current_low)
        threshold = SNN_FRACTION
        for _ in range(SNN_STEPS):
            low = (low >> decay) + current_low
            high = (high >> decay) + current_high
            check_signed(low, high, SNN_MEMBRANE_BITS, f"SNN layer {index} membrane")
            crosses = (low < threshold) & (high >= threshold)
            low = low - threshold * (low >= threshold)
            high = high - threshold * (high >= threshold)
            # Include both sides of the subtract-reset discontinuity.
            low = torch.where(crosses, low.clamp_max(0), low)
            high = torch.where(crosses, high.clamp_min(threshold - 1), high)

    def scores(self, images):
        """Read out spike counts and residual membrane as Q16.16 scores."""
        inputs = 255 - images.long()
        for index, (layer, decay) in enumerate(zip(self.model.layers, SNN_DECAYS)):
            current = apply_layer(inputs, layer, *self.parameters[index])
            if index == 0:
                currents = round_shift(current, SNN_STEM_SHIFT).unsqueeze(0).expand(SNN_STEPS, *current.shape)
            else:
                currents = current.reshape(SNN_STEPS, len(images), *current.shape[1:])
            membrane = torch.zeros_like(currents[0])
            spikes = []
            for step_current in currents:
                membrane = (membrane >> decay) + step_current
                fired = (membrane >= SNN_FRACTION).long()
                membrane = membrane - fired * SNN_FRACTION
                spikes.append(fired)
            inputs = torch.cat(spikes)
        units = torch.stack(spikes).sum(0) * SNN_FRACTION + membrane
        return (units << 18) // (SNN_STEPS * SNN_FRACTION)
