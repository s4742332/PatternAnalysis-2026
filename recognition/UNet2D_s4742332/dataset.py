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
1. Load scans from one folder 
2. Parse file names correctly
3. Patients and scans correctly connected in 1 folder
4. splits between folders
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass
import re
from collections import Counter 

DATA_ROOT = Path(r"S:\PatternAnalysis-2026\HipMRI_Study_Open\keras_slices_data")
_NAME_RE = re.compile(r"(?:case|seg)_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz")

# dataclasses
@dataclass
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

if __name__ == "__main__":
    print("Get jobs 1-4 working")
    files = list_folder( DATA_ROOT / "keras_slices_train")
    patients, scans = summarise_folder(DATA_ROOT/ "keras_slices_train")
    print(len(patients), len(scans))
    print(sorted(scans))
    patients, scans = summarise_folder(DATA_ROOT/ "keras_slices_seg_train")
    print(len(patients), len(scans))
    print(sorted(scans))
    patients, scans = summarise_folder(DATA_ROOT/ "keras_slices_validate")
    print(len(patients), len(scans))
    print(sorted(scans))
    patients, scans = summarise_folder(DATA_ROOT/ "keras_slices_seg_validate")
    print(len(patients), len(scans))
    print(sorted(scans))
    patients, scans = summarise_folder(DATA_ROOT/ "keras_slices_seg_test")
    print(len(patients), len(scans))
    print(sorted(scans))
    patients, scans = summarise_folder(DATA_ROOT/ "keras_slices_test")
    print(len(patients), len(scans))
    print(sorted(scans))
    print(len(files))
    for f in files[:5]:
        print(parse_name(f.name))