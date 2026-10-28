"""
Containing the source code for training, validating, testing, and saving your model.
The model should be imported from modules.py and the data loader should be imported from
dataset.py. Make sure to plot the losses and metrics during training.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import full_load_dataset, DATA_ROOT, CACHE_DIR, N_LABELS
from modules import UNet2D

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

if __name__ == "__main__":
    args = parse_args()
    device, loaders, model, loss_fn, optimiser, run_dir = setup(args)
