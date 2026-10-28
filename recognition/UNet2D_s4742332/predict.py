"""
Showing example usage of your trained model. It should load your saved model
weights, run inference on test cases, print out any results, and provide visualisations where applicable
(e.g., prediction overlays, generated samples, heatmaps). NumPy is allowed in this script for
visualization or loading data.
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import full_load_dataset, DATA_ROOT, CACHE_DIR, N_LABELS, H, W
from modules import UNet2D
from train import LABELS, REGIONS, prostate_regions

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained 2D U-Net on the HipMRI test split")
    p.add_argument("run_dirs", type=Path, nargs="+", help="run folders written by train.py")
    p.add_argument("--checkpoint", default="best.pt", help="checkpoint file inside each run folder")
    p.add_argument("--batch-size", type=int, default=16, help="slices per forward pass at inference")
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

def test_dataset(train_args):
    """the run's held-out test patients, rebuilt from its saved seed and split sizes, no augmentation"""
    loaders = full_load_dataset(seed=train_args["seed"], batch_size=1, augment=False, num_workers=0,
                                n_val=train_args["n_val"], n_test=train_args["n_test"], root=DATA_ROOT, cache_dir=CACHE_DIR)
    return loaders["test"].dataset

def cases_in(dataset):
    """(patient, week) -> dataset indices of that scan's slices, a case is one weekly scan"""
    cases = {}
    for k, sid in enumerate(dataset.ids):
        cases.setdefault((sid.patient, sid.week), []).append(k)
    return cases

def class_counts(pred, mask, n_classes=N_LABELS):
    """per-class overlap, predicted and true pixel counts, summed over the batch"""
    inter, pred_sum, true_sum = (torch.zeros(n_classes, device=mask.device) for _ in range(3))
    for c in range(n_classes):
        p, t = pred == c, mask == c
        inter[c], pred_sum[c], true_sum[c] = (p & t).sum(), p.sum(), t.sum()
    return inter, pred_sum, true_sum

def dice_iou(inter, pred_sum, true_sum, eps=1e-6):
    dice = (2 * inter + eps) / (pred_sum + true_sum + eps)
    iou = (inter + eps) / (pred_sum + true_sum - inter + eps)
    return dice.cpu(), iou.cpu()

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

@torch.no_grad()
def evaluate(model, dataset, device, batch_size):
    """Dice/IoU per class over the whole split, base/central/apex prostate Dice, per-case prostate Dice,
    peak VRAM and per-case inference time (upload + forward + argmax, data loading not counted)"""
    regions = prostate_regions(dataset)
    inter, pred_sum, true_sum = (torch.zeros(N_LABELS, device=device) for _ in range(3))
    reg_inter, reg_total = (torch.zeros(len(REGIONS), device=device) for _ in range(2))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model(torch.zeros(batch_size, 1, H, W, device=device))  # warm-up so cuDNN set-up isn't timed in the first case
    sync(device)

    cases = []
    for (patient, week), idx in cases_in(dataset).items():
        images, masks, names = zip(*(dataset[k] for k in idx))
        images, masks = torch.stack(images), torch.stack(masks)
        sync(device)
        t0 = time.perf_counter()
        pred = torch.cat([model(images[i:i + batch_size].to(device)).argmax(dim=1)
                          for i in range(0, len(idx), batch_size)])
        sync(device)
        seconds = time.perf_counter() - t0

        masks = masks.to(device)
        i, p, t = class_counts(pred, masks)
        inter += i
        pred_sum += p
        true_sum += t
        r = torch.tensor([regions.get(n, -1) for n in names], device=device)
        keep = r >= 0
        pred_p = (pred == LABELS["prostate"]).flatten(1)
        true_p = (masks == LABELS["prostate"]).flatten(1)
        reg_inter.index_add_(0, r[keep], (pred_p & true_p).sum(1)[keep].float())
        reg_total.index_add_(0, r[keep], (pred_p.sum(1) + true_p.sum(1))[keep].float())
        case_dice, _ = dice_iou(i, p, t)
        cases.append({"patient": patient, "week": week, "slices": len(idx), "seconds": seconds,
                      "prostate_dice": case_dice[LABELS["prostate"]].item()})

    dice, iou = dice_iou(inter, pred_sum, true_sum)
    region_dice = ((2 * reg_inter + 1e-6) / (reg_total + 1e-6)).cpu()
    peak_vram = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    return {"dice": dice.tolist(), "iou": iou.tolist(), "region_dice": region_dice.tolist(),
            "peak_vram_gb": peak_vram, "batch_size": batch_size, "cases": cases}

def print_results(name, results):
    seconds = np.array([c["seconds"] for c in results["cases"]])
    slices = np.array([c["slices"] for c in results["cases"]])
    case_dice = np.array([c["prostate_dice"] for c in results["cases"]])
    print(f"\n{name} test: {len(results['cases'])} cases, {slices.sum()} slices")
    print(f"  {'class':12} {'dice':>6} {'iou':>6}")
    for label, c in LABELS.items():
        print(f"  {label:12} {results['dice'][c]:6.3f} {results['iou'][c]:6.3f}")
    print(f"  prostate {' '.join(f'{r} {d:.3f}' for r, d in zip(REGIONS, results['region_dice']))}")
    print(f"  per-case prostate dice {case_dice.mean():.3f} +/- {case_dice.std():.3f} (min {case_dice.min():.3f})")
    print(f"  peak VRAM {results['peak_vram_gb']:.2f} GB, inference {1000 * seconds.mean():.0f} +/- {1000 * seconds.std():.0f} ms per case "
          f"({1000 * seconds.sum() / slices.sum():.1f} ms per slice, batch {results['batch_size']})")

def print_comparison(all_results):
    print(f"\n{'run':24} {'pros dice':>9} {'pros iou':>8} {'base':>6} {'central':>7} {'apex':>6} {'case dice':>13} {'VRAM GB':>7} {'ms/case':>7}")
    for name, r in all_results.items():
        case_dice = np.array([c["prostate_dice"] for c in r["cases"]])
        ms = 1000 * np.mean([c["seconds"] for c in r["cases"]])
        pr = LABELS["prostate"]
        print(f"{name:24} {r['dice'][pr]:9.3f} {r['iou'][pr]:8.3f} "
              f"{r['region_dice'][0]:6.3f} {r['region_dice'][1]:7.3f} {r['region_dice'][2]:6.3f} "
              f"{case_dice.mean():6.3f}+/-{case_dice.std():.3f} {r['peak_vram_gb']:7.2f} {ms:7.0f}")

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets, all_results = {}, {}
    for run_dir in args.run_dirs:
        model, train_args, ckpt = load_model(run_dir / args.checkpoint, device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{run_dir.name}: epoch {ckpt['epoch']}, {n_params:.2f}M params, "
              f"skip {train_args['skip_connected']}, depths {train_args['conv_depths']}, "
              f"seed {train_args['seed']}, val prostate dice {ckpt['val_dice'][LABELS['prostate']]:.3f} "
              f"({' '.join(f'{name} {d:.3f}' for name, d in zip(REGIONS, val_region_dice(run_dir, ckpt['epoch'])))})")

        split = (train_args["seed"], train_args["n_val"], train_args["n_test"])
        if split not in datasets:  # runs with the same split share the test set
            datasets[split] = test_dataset(train_args)
        results = evaluate(model, datasets[split], device, args.batch_size)
        print_results(run_dir.name, results)
        with open(run_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=1)
        all_results[run_dir.name] = results
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()  # so the next run's peak VRAM is its own
    if len(all_results) > 1:
        print_comparison(all_results)
