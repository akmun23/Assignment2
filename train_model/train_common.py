"""Data loading, training loop and checkpoint files shared by the four scripts."""

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from augment_data import CLASSES, augment, contributor, image_files, load_images


RUNS = Path(__file__).resolve().parent / "runs"
CHECKPOINTS = RUNS.parent / "checkpoints"
TEACHER_DIR = CHECKPOINTS / "teacher"
CORPORA = ("hand", "synthetic")


# ---------------------- Data ----------------------
class Data:
    def __init__(self, device, excluded=None):
        self.device = device
        self.train = {}
        self.validation = {}
        hand_groups = {}

        for corpus in CORPORA:
            for part, split in (("train", self.train), ("validation", self.validation)):
                paths, labels = image_files(corpus, part)
                if corpus == "synthetic" and part == "validation":
                    rng = np.random.default_rng(20261118)
                    selected = np.sort(np.concatenate([
                        rng.choice(np.flatnonzero(labels == c), 3000, replace=False)
                        for c in range(len(CLASSES))
                    ]))
                    paths, labels = [paths[i] for i in selected], labels[selected]
                elif excluded is not None and part == "train":
                    selected = np.flatnonzero(labels != excluded)
                    paths, labels = [paths[i] for i in selected], labels[selected]
                if corpus == "hand":
                    hand_groups[part] = [contributor(path) for path in paths]
                images = load_images(paths)
                split[corpus] = (torch.from_numpy(images).unsqueeze(1).to(device),
                                 torch.from_numpy(labels).to(device))

        if not set(hand_groups["train"]).isdisjoint(hand_groups["validation"]):
            raise ValueError("Handwritten contributors overlap between training and validation")
        hand_labels = self.train["hand"][1].cpu().numpy()
        _, group_index, group_sizes = np.unique(hand_groups["train"], return_inverse=True,
                                                return_counts=True)
        weights = group_sizes[group_index] ** -0.5
        for label in np.unique(hand_labels):
            keep = hand_labels == label
            weights[keep] /= weights[keep].sum()
        self.hand_weights = torch.from_numpy(weights / weights.sum()).float().to(device)

    def batches(self, samples, batch_size, generator):
        hand_images, hand_labels = self.train["hand"]
        syn_images, syn_labels = self.train["synthetic"]
        hand_count = int(samples * 0.7)
        hand = torch.multinomial(self.hand_weights, hand_count, replacement=True, generator=generator)
        synthetic = torch.randint(len(syn_labels), (samples - hand_count,),
                                  device=self.device, generator=generator)
        indices = torch.cat([hand, synthetic])
        source = torch.arange(samples, device=self.device) >= hand_count
        order = torch.randperm(samples, device=self.device, generator=generator)
        for start in range(0, samples, batch_size):
            selection = order[start:start + batch_size]
            index, is_syn = indices[selection], source[selection]
            h = index.clamp_max(len(hand_labels) - 1)
            s = index.clamp_max(len(syn_labels) - 1)
            images = torch.where(is_syn[:, None, None, None], syn_images[s], hand_images[h])
            labels = torch.where(is_syn, syn_labels[s], hand_labels[h])
            yield augment(images.float() / 255), labels


# ---------------------- Training ----------------------
def setup():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    print(f"Training on {device}", flush=True)
    return device


def classification_loss(logits, labels, teacher_logits):
    classification = F.cross_entropy(logits, labels, label_smoothing=0.03)
    temperature = 3
    distillation = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                           F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean")
    return 0.5 * classification + 0.5 * temperature**2 * distillation


@torch.inference_mode()
def predict(model, images, batch_size=2048):
    model.eval()
    return torch.cat([model(batch.float() / 255) for batch in images.split(batch_size)])


def classifier_metrics(logits, labels):
    top = logits.argsort(dim=1, descending=True)
    return dict(n=len(labels), acc=(top[:, 0] == labels).float().mean().item(),
                top2=(top[:, :2] == labels[:, None]).any(dim=1).float().mean().item())


def train(model, data, loss_function, metrics, *, epochs, samples, batch_size, lr, seed,
          validation_every=5, select_best=False):
    generator = torch.Generator(device=data.device).manual_seed(seed)
    average = deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    steps = epochs * math.ceil(samples / batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps,
                                                  pct_start=0.12, div_factor=15, final_div_factor=200)
    best_score = -1
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for images, labels in data.batches(samples, batch_size, generator):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), images, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for running, current in zip(average.state_dict().values(), model.state_dict().values()):
                    if running.is_floating_point():
                        running.lerp_(current, 0.002)
                    else:
                        running.copy_(current)
            total_loss += loss.item() * len(labels)

        if epoch % validation_every == 0 or epoch == epochs:
            row = dict(epoch=epoch, loss=total_loss / samples)
            for corpus, (images, labels) in data.validation.items():
                row[corpus] = metrics(predict(average, images, batch_size), labels)
            history.append(row)
            print(json.dumps(row), flush=True)
            if select_best:
                score = row["hand"]["acc"] + 0.05 * row["synthetic"]["acc"]
                if score > best_score:
                    best_score = score
                    best_model, best_epoch = deepcopy(average), epoch

    return (best_model, history, best_epoch) if select_best else (average, history, epochs)


# ---------------------- Saving ----------------------
def save_model(folder, weights, history, data, *, logits, **metadata):
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(weights, folder / ("head.pt" if metadata["role"] == "pair" else "model.pt"))
    np.savez_compressed(folder / "val_logits.npz",
                        hand=logits["hand"].cpu().numpy(), synthetic=logits["synthetic"].cpu().numpy(),
                        hand_y=data.validation["hand"][1].cpu().numpy(),
                        synthetic_y=data.validation["synthetic"][1].cpu().numpy())
    (folder / "metrics.json").write_text(json.dumps(dict(history=history, **metadata), indent=2) + "\n")
