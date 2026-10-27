"""
Containing the data loader for loading and preprocessing your data, including leakage-
free train/validation/test splitting functions and data augmentations
"""

"""
Data Notes:
1. 
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import re
from collections import Counter 
from itertools import combinations
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

DATA_ROOT = Path(r"S:\PatternAnalysis-2026\HipMRI_Study_Open\keras_slices_data")
_NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")
SPLITS = ("train", "validate", "test")
SPLITS = ("train", "validate", "test")

# dataclasses
@dataclass(frozen=True)
class SliceID:
    """one slice: patient case, weekly scan, axial slice index."""
    case: str
    week: int
    slice: int

# folder work

def parse_name(name: str) -> SliceID:
    m = _NAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"bad filename: {name}")
    return SliceID(
        case=f"{int(m.group(1)):03d}",
        week=int(m.group(2)),    
        slice=int(m.group(3)),    
    )

def list_folder(folder: Path) -> list[Path]:
    return sorted(folder.iterdir())

def summarise_folder(folder: Path):
    patients = set()
    scans = set()
    slices_per_scan = Counter()

    for f in folder.iterdir():
        sid = parse_name(f.name)
        patients.add(sid.case)
        scans.add((sid.case, sid.week)) # a scan = week + case
        slices_per_scan[(sid.case, sid.week)] += 1

    # every scan should have 60 slices 
    bad = check_slice_count(slices_per_scan)
    assert not bad, f"scans without 60 slices: {bad}"
    
    return patients, scans

def ids_in(folder: Path) -> set[SliceID]:
    return {parse_name(f.name) for f in folder.iterdir()}

def patients_by_split(root: Path):
    result = {}
    for split in SPLITS:
        patients, scans = summarise_folder(root / f"keras_slices_{split}")
        print(f"{split:9} patients={len(patients):3}  scans={len(scans):4}")
        result[split] = patients    # TODO: what goes in the dict?
    return result

# data checks/investigation -> class soon?
def check_pairs(root: Path, split: str) -> None:
    images = ids_in(root / f"keras_slices_{split}")
    masks  = ids_in(root / f"keras_slices_seg_{split}")
    only_images = images - masks     # set difference
    only_masks  = masks - images
    assert not only_images and not only_masks, (
        f"{split}: {len(only_images)} images without masks, "
        f"{len(only_masks)} masks without images"
    )
    print(f"{split}: {len(images)} images/masks paired OK")

def check_no_patient_leakages(patients: dict[str, set[str]]):
    problems = []
    for a, b in combinations(patients, 2):
        shared = patients[a] & patients[b]          # patients in both splits
        if shared:
            problems.append(f"{a} & {b} share {sorted(shared)}")
    if problems:
        raise ValueError("patient leakage: " + "; ".join(problems))
    print("no leakage")

def check_slice_count(slices_per_scan):
    return {scan: n for scan, n in slices_per_scan.items() if n!= 60}

def check_labels(root:Path):
    shapes, dtypes, zooms, labels = Counter(), Counter(), Counter(), Counter()
    for f in root.iterdir():
        img = nib.load(f)
        shapes[img.shape] += 1
        zooms[tuple(round(z, 3) for z in img.header.get_zooms())] += 1
        dtypes[str(img.get_data_dtype())] += 1
        if "seg" in root.name:
            v, c = np.unique(np.asarray(img.dataobj), return_counts=True)
            for a, b in zip(v, c): labels[int(a)] += int(b)
    print(root.name, dict(shapes), dict(dtypes), dict(zooms))
    if labels:
        tot = sum(labels.values())
        print("   labels:", {k: f"{v/tot:.4%}" for k, v in sorted(labels.items())})
    return

def check_images_manually(root:Path, split: str, sid: SliceID):
    img_path, seg_path = sliceId_to_dir(root=root, split=split, sid=sid)
    image = nib.load(img_path).get_fdata()
    mask = np.asarray(nib.load(seg_path).dataobj)
    labels = np.unique(mask)
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 5))
    for ax, k in zip(axes, labels):
        ax.imshow(image.T, cmap="gray")                                  # .T: see note below
        ax.imshow(np.ma.masked_where(mask != k, mask).T, alpha=0.5, cmap="autumn")
        ax.set_title(f"label {k}  ({(mask == k).mean():.2%})")
        ax.axis("off")
    fig.suptitle(f"case {sid.case} week {sid.week} slice {sid.slice}")
    plt.show()

def check_weeks_per_case(root:Path, split:str):
    images = ids_in(root / f"keras_slices_{split}")
    weeks_per_case = {}
    for i in images:
        weeks_per_case.setdefault(i.case, set())
        weeks_per_case[i.case].add(i.week)

    for case, weeks in sorted(weeks_per_case.items() ):
        print(f"{split} case {case}: {len(weeks)} weeks {sorted(weeks)}")
    return weeks_per_case


class HipMRISlices(Dataset):


if __name__ == "__main__":
    results = patients_by_split( DATA_ROOT)
    #check_images_manually(DATA_ROOT, "train", SliceID(case="004", week=0, slice=45))
    for split in SPLITS:
        #check_pairs(DATA_ROOT, split)
        #check_no_patient_leakages(results)
        #check_labels(DATA_ROOT / f"keras_slices_seg_{split}")
        _ = check_weeks_per_case(DATA_ROOT, split)
        #check_images_manually(DATA_ROOT, f"keras_slices_seg_{split}", SliceID(case="004", week=0, slice=30))
