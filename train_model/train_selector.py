#!/usr/bin/env python3
"""Train the selector and its three pair heads together."""

import torch
from torch import nn
import torch.nn.functional as F

from augment_data import CLASSES
from train_cnn import CNN, load_cnn
from train_common import RUNS, TEACHER_DIR, Data, classifier_metrics, predict, save_model, setup, train
from train_head import Head, PAIRS, pair_loss, save_head


# ===================== EDIT THESE =====================
OUTPUT_DIR = RUNS / "selector"
TEACHER = TEACHER_DIR
SEED = 1
EPOCHS = 120
SAMPLES = 30_000
BATCH = 256
LR = 4e-3


# ---------------------- Model ----------------------
class Cascade(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector = CNN(stem=6, first=10, second=12, hidden=16, kernel=4, stride=4)
        self.heads = nn.ModuleList(Head() for _ in CLASSES)

    def forward(self, images):
        features = self.selector.features(images)
        scores = self.selector.classify(features)
        return torch.column_stack([scores, *(head(features) for head in self.heads)])


# ---------------------- Loss and validation ----------------------
def cascade_loss(logits, labels, teacher_probs):
    scores, heads = logits.split(len(CLASSES), dim=1)
    excluded_prob = F.softmax(-scores, dim=1).gather(1, labels[:, None]).squeeze(1)
    # Only penalise the selector for putting the true class last.
    loss = -torch.log1p(-excluded_prob.clamp(max=1 - 1e-6)).mean()
    for excluded, head_scores in enumerate(heads.unbind(dim=1)):
        loss = loss + pair_loss(head_scores, labels, excluded, teacher_probs)
    return loss


def cascade_metrics(logits, labels):
    scores, heads = logits.split(len(CLASSES), dim=1)
    excluded = scores.argmin(dim=1)
    pair = scores.new_tensor(PAIRS, dtype=torch.long)[excluded]
    head_scores = heads.gather(1, excluded[:, None]).squeeze(1)
    decisions = torch.where(head_scores >= 0, pair[:, 0], pair[:, 1])
    return dict(**classifier_metrics(scores, labels),
                cascade=(decisions == labels).float().mean().item())


# ---------------------- Main ----------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    device = setup()
    teacher = load_cnn(TEACHER, device)
    torch.manual_seed(SEED)
    model = Cascade().to(device)
    data = Data(device)

    def loss(logits, images, labels):
        return cascade_loss(logits, labels, teacher(images).softmax(dim=1))

    model, history, _ = train(model, data, loss, cascade_metrics, epochs=EPOCHS, samples=SAMPLES,
                              batch_size=BATCH, lr=LR, seed=SEED, validation_every=10)
    logits = {name: predict(model, images) for name, (images, _) in data.validation.items()}
    selector_dir = OUTPUT_DIR / "selector"
    save_model(selector_dir, model.selector.state_dict(), history, data, role="selector", cls=-1,
               spec=model.selector.spec, final=history[-1],
               logits={name: scores[:, :len(CLASSES)] for name, scores in logits.items()})
    for excluded, shape in enumerate(CLASSES):
        head_logits = {name: scores[:, len(CLASSES) + excluded] for name, scores in logits.items()}
        save_head(OUTPUT_DIR / f"head_{shape}", model.heads[excluded], excluded,
                  selector_dir, data, head_logits, history)
    print(f"Saved selector and heads to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
