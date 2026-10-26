"""
Containing the data loader for loading and preprocessing your data, including leakage-
free train/validation/test splitting functions and data augmentations
"""

"""
Data Notes:
1. presegemented into test, train, validate
2. seg_XXX_week_X_slice_X.nii.gz, case_XXX_week_X_slice_XX.nii,gz
    - Assume Patient, XXXX, and 1 scan is 1 pateint + week
    - supposed to be 38 patients, and 211 scans total
3. Data must be of same size, and greyscale

JOBS:
1. Determine which patients appear in other splits, we don't want to be using seen training data for testing/validation
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import re
from collections import Counter 
from itertools import combinations

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
    bad = {scan: n for scan, n in slices_per_scan.items() if n!= 60}
    assert not bad, f"scans without 60 slices: {bad}"
    
    return patients, scans

def ids_in(folder: Path) -> set[SliceID]:
    return {parse_name(f.name) for f in folder.iterdir()}

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


def patients_by_split(root: Path) -> dict[str, set[str]]:
    result = {}
    for split in SPLITS:
        patients, scans = summarise_folder(root / f"keras_slices_{split}")
        print(f"{split:9} patients={len(patients):3}  scans={len(scans):4}")
        result[split] = patients    # TODO: what goes in the dict?
    return result

def check_no_patient_leakages(patients: dict[str, set[str]]):
    problems = []
    for a, b in combinations(patients, 2):
        shared = patients[a] & patients[b]          # patients in both splits
        if shared:
            problems.append(f"{a} & {b} share {sorted(shared)}")
    if problems:
        raise ValueError("patient leakage: " + "; ".join(problems))
    print("no leakage")


if __name__ == "__main__":
    print("Get jobs 1-4 working")
    results = patients_by_split( DATA_ROOT)
    for split in SPLITS:
        check_pairs(DATA_ROOT, split)
        results = patients_by_split( DATA_ROOT )
        check_no_patient_leakages(results)
    
    good = {"train": {"004", "005"}, "validate": {"036"}, "test": {"040"}}
    bad  = {"train": {"004", "040"}, "validate": {"036"}, "test": {"040"}}
    check_no_patient_leakages(good)   # prints: no leakage
    check_no_patient_leakages(bad)    # crashes: ValueError: patient leakage: train & test share ['040']
