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
import numpy as np, random
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Optional

from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

DATA_ROOT = Path(r"S:\PatternAnalysis-2026\HipMRI_Study_Open\keras_slices_data")
_NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")
SPLITS = ("train", "validate", "test")


# dataclasses
@dataclass(frozen=True, order=True)
class SliceID:
    """one slice: patient case, weekly scan, axial slice index and the paths to both it's image and mask paths
    not a seperated ID as mask-images must always be paired
    """
    patient: int # parsed as int so 004 and 4 are the same patient
    week: int
    slice: int
    image_path: Path
    mask_path: Path

# folder work
def name_to_paths(root: Path, split: str, name: str):
    """image file name -> (image path, mask path). mask name is the image name with case_ swapped for seg_"""
    mask_name = "seg_" + name.removeprefix("case_")
    return root / f"keras_slices_{split}" / name, root / f"keras_slices_seg_{split}" / mask_name

def resplit_patients(all_slice_ids: dict[str, list[SliceID]], seed: int | None = None,
                     n_val: int = 4, n_test: int = 3):
    if seed is None:
        return all_slice_ids                        # default: keep the provided split

    pooled = [s for ids in all_slice_ids.values() for s in ids]
    patients = sorted(patients_in(pooled))          # fixed start order -> same seed, same split
    if n_val + n_test >= len(patients):
        raise ValueError(f"n_val + n_test must be < {len(patients)} patients")

    patients = np.random.default_rng(seed).permutation(patients)
    test_p  = patients[:n_test]
    val_p   = patients[n_test:n_test + n_val]
    train_p = patients[n_test + n_val:]

    split_of = {}
    for p in train_p: split_of[int(p)] = "train"
    for p in val_p:   split_of[int(p)] = "validate"
    for p in test_p:  split_of[int(p)] = "test"

    new = {split: [] for split in SPLITS}
    for s in pooled:
        new[split_of[s.patient]].append(s)
    new = {split: sorted(ids) for split, ids in new.items()}

    check_no_patient_leakages({split: patients_in(ids) for split, ids in new.items()})
    return new

def build_sliceids(root: Path, split: str):
    check_pairs(root=root, split=split)
    all_ids = []
    for name in file_names_in(root / f"keras_slices_{split}"):
        img_path, mask_path = name_to_paths(root, split, name)
        all_ids.append(parse_to_sliceid(name, img_path=img_path, mask_path=mask_path))
    return sorted(all_ids) # file_names_in is a set, so sort by patient, week, slice

def build_all_sliceids(root: Path):
    all_slice_ids = {split: build_sliceids(root, split) for split in SPLITS}
    check_no_patient_leakages({split: patients_in(ids) for split, ids in all_slice_ids.items()})
    for split, ids in all_slice_ids.items():
        bad = check_slice_count(Counter((s.patient, s.week) for s in ids))
        if bad:
            raise ValueError(f"{split}: scans without 60 slices: {bad}")
    return all_slice_ids

def parse_file_name(name: str):
    m = _NAME_RE.fullmatch(name)
    if m is None:
        raise ValueError(f"bad filename: {name}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)) #patient, week, slice

def parse_to_sliceid(name: str, img_path: Path, mask_path: Path) -> SliceID:
    patient, week, slice_idx = parse_file_name(name=name)
    return SliceID(
        patient=patient,
        week=week,
        slice=slice_idx,
        image_path=img_path,
        mask_path=mask_path
    )

def file_names_in(folder: Path):
    return {f.name for f in folder.iterdir()}

def patients_in(ids: list[SliceID]):
    return {s.patient for s in ids}

def summarise_split(split: str, ids: list[SliceID]):
    scans = {(s.patient, s.week) for s in ids} # a scan = patient + week
    print(f"{split:9} patients={len(patients_in(ids)):3}  scans={len(scans):4}  slices={len(ids):5}")

# data checks/investigation -> class soon?
def check_pairs(root: Path, split: str):
    # compare parsed (patient, week, slice) tuples, the names themselves differ (case_ vs seg_)
    images = {parse_file_name(n) for n in file_names_in(root / f"keras_slices_{split}")}
    masks  = {parse_file_name(n) for n in file_names_in(root / f"keras_slices_seg_{split}")}
    only_images = images - masks     # set difference
    only_masks  = masks - images
    if only_images or only_masks:
        raise ValueError(
            f"{split}: {len(only_images)} images without masks, "
            f"{len(only_masks)} masks without images"
        )
    print(f"{split}: {len(images)} images/masks paired OK")

def check_no_patient_leakages(patients: dict[str, set[int]]):
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

def check_images_manually(sid: SliceID):
    image = nib.load(sid.image_path).get_fdata()
    mask = np.asarray(nib.load(sid.mask_path).dataobj)
    labels = np.unique(mask)
    fig, axes = plt.subplots(1, len(labels), figsize=(3 * len(labels), 5))
    for ax, k in zip(axes, labels):
        ax.imshow(image.T, cmap="gray")                                  # .T: see note below
        ax.imshow(np.ma.masked_where(mask != k, mask).T, alpha=0.5, cmap="autumn")
        ax.set_title(f"label {k}  ({(mask == k).mean():.2%})")
        ax.axis("off")
    fig.suptitle(f"patient {sid.patient:03d} week {sid.week} slice {sid.slice}")
    plt.show()

def check_weeks_per_case(split: str, ids: list[SliceID]):
    weeks_per_case = {}
    for s in ids:
        weeks_per_case.setdefault(s.patient, set())
        weeks_per_case[s.patient].add(s.week)

    for patient, weeks in sorted(weeks_per_case.items() ):
        print(f"{split} patient {patient:03d}: {len(weeks)} weeks {sorted(weeks)}")
    return weeks_per_case


class HipMRISlices(Dataset):
    """
    2D slices
    """
    def __init__(self):
        return

if __name__ == "__main__":
    all_ids = build_all_sliceids(DATA_ROOT)
    all_ids = resplit_patients(all_ids, 1, 5, 5)
    
    for split, ids in all_ids.items():
        summarise_split(split, ids)
        print("   first:", ids[:3])
    
    
        #_ = check_weeks_per_case(split, ids)
        #check_labels(DATA_ROOT / f"keras_slices_seg_{split}")
    #check_images_manually(all_ids["train"][45])
