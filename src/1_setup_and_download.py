"""
Setup and Data Download
Author: Amitabh Chakravorty

Downloads DS2OS and NSL-KDD datasets from Kaggle, creates the project directory
structure, and performs a quick sanity check on each dataset.

Usage:
    python src/1_setup_and_download.py --base_path ./

Prerequisites:
    - Kaggle API credentials at ~/.kaggle/kaggle.json
    - pip install kaggle pandas numpy
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and inspect datasets for cryptojacking validation"
    )
    parser.add_argument(
        "--base_path", type=str, default=".",
        help="Root directory for the project (default: current directory)"
    )
    return parser.parse_args()


def create_directories(base):
    """Build the project directory tree."""
    subdirs = [
        "data/raw", "data/processed",
        "models",
        "results", "results/metrics", "results/figures",
        "scripts",
    ]
    for sub in subdirs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    print(f"Project directories verified at: {base}")


def download_from_kaggle(dataset_id, label, raw_dir):
    """Pull a single Kaggle dataset and unzip it."""
    dest = os.path.join(raw_dir, label)
    os.makedirs(dest, exist_ok=True)

    print(f"  Downloading {label} ({dataset_id})")
    os.system(f'kaggle datasets download -d {dataset_id} -p "{dest}" --unzip')

    files = [f for f in glob.glob(os.path.join(dest, "**", "*"), recursive=True)
             if os.path.isfile(f)]
    print(f"  {label}: {len(files)} file(s) found")
    return len(files)


def quick_look(filepath, name, sample_rows=1000):
    """Read a small sample and print basic statistics."""
    print(f"\n--- {name} ---")
    print(f"File: {filepath}")

    df = None
    for enc in ("utf-8", "latin-1", "iso-8859-1"):
        try:
            df = pd.read_csv(filepath, encoding=enc, nrows=sample_rows)
            break
        except Exception:
            continue

    if df is None:
        print("  Could not load with common encodings.")
        return

    print(f"  Shape (sample): {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    missing_total = df.isnull().sum().sum()
    print(f"  Missing values in sample: {missing_total}")

    # Look for likely target columns
    hints = ["label", "class", "target", "attack", "type", "category"]
    candidates = [c for c in df.columns if any(h in c.lower() for h in hints)]
    if candidates:
        print(f"  Probable target column(s): {candidates}")
        for col in candidates[:2]:
            print(f"\n  Distribution of '{col}':")
            print(df[col].value_counts(dropna=False).head(15).to_string())


def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)

    print("CRYPTOJACKING VALIDATION - SETUP")
    print(f"Python {sys.version.split()[0]}\n")

    # Step 1 - directories
    print("[1/3] Creating project structure")
    create_directories(base)

    # Step 2 - download
    print("\n[2/3] Downloading datasets")
    kaggle_sets = {
        "ds2os":   "libamariyam/ds2os-dataset",
        "nsl_kdd": "hassan06/nslkdd",
    }
    raw_dir = os.path.join(base, "data", "raw")
    for label, kid in kaggle_sets.items():
        download_from_kaggle(kid, label, raw_dir)

    # Step 3 - quick exploration
    print("\n[3/3] Quick data check")

    ds2os_csvs = glob.glob(os.path.join(raw_dir, "ds2os", "**", "*.csv"), recursive=True)
    if ds2os_csvs:
        quick_look(ds2os_csvs[0], "DS2OS")

    nsl_all = (glob.glob(os.path.join(raw_dir, "nsl_kdd", "**", "*.csv"), recursive=True)
               + glob.glob(os.path.join(raw_dir, "nsl_kdd", "**", "*.txt"), recursive=True))
    train_file = next((f for f in nsl_all if "train" in os.path.basename(f).lower()), None)
    if train_file:
        quick_look(train_file, "NSL-KDD")
    elif nsl_all:
        quick_look(nsl_all[0], "NSL-KDD")

    # Persist a small record of what was downloaded
    info_path = os.path.join(base, "results", "setup_info.txt")
    with open(info_path, "w") as fh:
        fh.write(f"timestamp: {pd.Timestamp.now()}\n")
        fh.write(f"datasets: {list(kaggle_sets.keys())}\n")
        fh.write(f"base_path: {base}\n")
        fh.write(f"python: {sys.version.replace(chr(10), ' ')}\n")

    print(f"\nSetup info written to {info_path}")
    print("DONE")


if __name__ == "__main__":
    main()
