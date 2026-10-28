"""
Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without a window
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dataset import full_load_dataset, DATA_ROOT, CACHE_DIR, N_LABELS
from modules import UNet2D

LABEL_NAMES = ("background", "body", "femoral hip", "bladder", "rectum", "prostate")
PROSTATE = 5

def parse_args():
    p = argparse.ArgumentParser(description="Train 2D U-Net on HipMRI prostate slices")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--n-val", type=int, default=5)
    p.add_argument("--n-test", type=int, default=5)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--no-skip", dest="skip_connected", action="store_false",
                   help="baseline model without skip connections")
    p.add_argument("--patient-weighting", choices=["sqrt", "inverse"], default=None)
    p.add_argument("--conv-depths", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    loaders = full_load_dataset(seed=args.seed, batch_size=args.batch_size, augment=args.augment,
                                num_workers=args.num_workers, n_val=args.n_val, n_test=args.n_test,
                                root=DATA_ROOT, cache_dir=CACHE_DIR, patient_weighting=args.patient_weighting)

    model = UNet2D(1, N_LABELS, conv_depths=tuple(args.conv_depths), skip_connected=args.skip_connected).to(device)
    loss_fn = nn.CrossEntropyLoss() 
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    run_name = f"{'skip' if args.skip_connected else 'noskip'}_pw-{args.patient_weighting}_seed{args.seed}"
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{run_name} on {device}, {n_params:.2f}M params, "
          + ", ".join(f"{s}: {len(l)} batches" for s, l in loaders.items()))
    return device, loaders, model, loss_fn, optimiser, run_dir

def dice_counts(logits, mask, n_classes=N_LABELS):
    pred = logits.argmax(dim=1)
    inter = torch.zeros(n_classes, device=mask.device)
    total = torch.zeros(n_classes, device=mask.device)
    for c in range(n_classes):
        p, t = pred == c, mask == c
        inter[c] = (p & t).sum()
        total[c] = p.sum() + t.sum()
    return inter, total

def dice_from_counts(inter, total, eps=1e-6):
    return ((2 * inter + eps) / (total + eps)).cpu()

def run_epoch(model, loader, loss_fn, device, optimiser=None):
    training = optimiser is not None
    model.train(training)
    loss_sum, n_seen = 0.0, 0
    inter = torch.zeros(N_LABELS, device=device)
    total = torch.zeros(N_LABELS, device=device)
    with torch.set_grad_enabled(training):
        for image, mask, _ in loader:
            image = image.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(image)
            loss = loss_fn(logits, mask)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            loss_sum += loss.item() * image.shape[0]
            n_seen += image.shape[0]
            i, t = dice_counts(logits.detach(), mask)
            inter += i
            total += t
    return loss_sum / n_seen, dice_from_counts(inter, total)

def plot_history(history, run_dir):
    fig, (ax_loss, ax_dice) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validate")
    ax_loss.set(xlabel="epoch", ylabel="cross entropy", title="Loss")
    ax_loss.legend()
    val_dice = np.array(history["val_dice"])
    for c in range(N_LABELS):
        ax_dice.plot(epochs, val_dice[:, c], label=LABEL_NAMES[c], linewidth=2.5 if c == PROSTATE else 1)
    ax_dice.axhline(0.75, color="grey", linestyle="--", linewidth=0.8)
    ax_dice.set(xlabel="epoch", ylabel="dice", title="Validation Dice", ylim=(0, 1))
    ax_dice.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=120)
    plt.close(fig)

def train(args, device, loaders, model, loss_fn, optimiser, run_dir):
    history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}
    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_dice = run_epoch(model, loaders["train"], loss_fn, device, optimiser)
        val_loss, val_dice = run_epoch(model, loaders["validate"], loss_fn, device)
        for key, value in zip(history, (train_loss, val_loss, train_dice.tolist(), val_dice.tolist())):
            history[key].append(value)

        prostate = val_dice[PROSTATE].item()
        improved = prostate > best_dice
        if improved:
            best_dice = prostate
            torch.save({"epoch": epoch, "model": model.state_dict(), "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "val_dice": val_dice.tolist()},
                       run_dir / "best.pt")
        torch.save({"epoch": epoch, "model": model.state_dict(), "optimiser": optimiser.state_dict()}, run_dir / "last.pt")

        print(f"epoch {epoch:3d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val dice {' '.join(f'{name} {d:.3f}' for name, d in zip(LABEL_NAMES, val_dice.tolist()))}"
              f"{' *' if improved else ''} | {time.time() - t0:.0f}s")
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=1)
        plot_history(history, run_dir)
    print(f"best val prostate dice {best_dice:.4f}, saved to {run_dir / 'best.pt'}")
    return history

if __name__ == "__main__":
    args = parse_args()
    device, loaders, model, loss_fn, optimiser, run_dir = setup(args)
    train(args, device, loaders, model, loss_fn, optimiser, run_dir)
