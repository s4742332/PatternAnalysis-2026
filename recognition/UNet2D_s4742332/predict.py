"""
Showing example usage of your trained model. It should load your saved model
weights, run inference on test cases, print out any results, and provide visualisations where applicable
(e.g., prediction overlays, generated samples, heatmaps). NumPy is allowed in this script for
visualization or loading data.
"""

import argparse
import glob
import json
from pathlib import Path

import torch

from dataset import N_LABELS
from modules import UNet2D
from train import LABELS, REGIONS

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained 2D U-Net on the HipMRI test split")
    p.add_argument("run_dirs", type=Path, nargs="+", help="run folders written by train.py")
    p.add_argument("--checkpoint", default="best.pt", help="checkpoint file inside each run folder")
    args = p.parse_args()
    # PowerShell passes wildcards through unexpanded, so expand runs/* here
    run_dirs = []
    for pattern in args.run_dirs:
        matches = sorted(Path().glob(str(pattern))) if glob.has_magic(str(pattern)) else [pattern]
        run_dirs.extend(d for d in matches if d.is_dir())
        if not any(d.is_dir() for d in matches):
            p.error(f"no run folder matches {pattern} (relative to {Path.cwd()})")
    args.run_dirs = run_dirs
    return args

def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "args" not in ckpt:
        raise ValueError(f"{checkpoint_path} has no saved args, use best.pt rather than last.pt")
    args = ckpt["args"]
    model = UNet2D(1, N_LABELS, conv_depths=tuple(args["conv_depths"]), skip_connected=args["skip_connected"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, args, ckpt

def val_region_dice(run_dir, epoch):
    """base/central/apex val prostate dice at the checkpoint's epoch, from train.py's history.json"""
    with open(run_dir / "history.json") as f:
        return json.load(f)["val_region_dice"][epoch - 1]

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run_dir in args.run_dirs:
        model, train_args, ckpt = load_model(run_dir / args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{run_dir.name}: epoch {ckpt['epoch']}, {n_params:.2f}M params, "
              f"skip {train_args['skip_connected']}, depths {train_args['conv_depths']}, "
              f"seed {train_args['seed']}, val prostate dice {ckpt['val_dice'][LABELS['prostate']]:.3f} "
              f"({' '.join(f'{name} {d:.3f}' for name, d in zip(REGIONS, val_region_dice(run_dir, ckpt['epoch'])))})")
