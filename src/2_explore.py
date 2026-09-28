"""
Data Exploration
Author: Amitabh Chakravorty

Loads the raw DS2OS and NSL-KDD datasets, prints shape, types, class
distributions, and saves distribution bar charts.

Usage:
    python src/2_explore.py --base_path ./
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Explore downloaded datasets")
    parser.add_argument("--base_path", type=str, default=".",
                        help="Project root (default: current directory)")
    return parser.parse_args()


# ---- DS2OS ----------------------------------------------------------------

def explore_ds2os(base):
    """Examine the DS2OS CSV: shape, types, target distribution, and a bar chart."""
    ds2os_dir = os.path.join(base, "data", "raw", "ds2os")
    csvs = [f for f in os.listdir(ds2os_dir) if f.endswith(".csv")]

    if not csvs:
        print("No CSV files found for DS2OS")
        return None, []

    print("DS2OS DATASET")
    filepath = os.path.join(ds2os_dir, csvs[0])
    df = pd.read_csv(filepath)

    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    print(f"\n  Data types:\n{df.dtypes.value_counts().to_string()}")

    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n  Missing values:\n{missing[missing > 0].to_string()}")
    else:
        print("  No missing values")

    # Identify target columns
    keywords = ["label", "class", "target", "attack", "type", "category"]
    targets = [c for c in df.columns if any(k in c.lower() for k in keywords)]

    if targets:
        print(f"\n  Target column candidates: {targets}")
        for col in targets:
            print(f"\n  '{col}' value counts:")
            print(df[col].value_counts().to_string())

            fig, ax = plt.subplots(figsize=(10, 4))
            df[col].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"DS2OS - Distribution of '{col}'")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            out = os.path.join(base, "results", "figures", f"ds2os_{col}_distribution.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Chart saved: {out}")

    return csvs[0], targets


# ---- NSL-KDD --------------------------------------------------------------

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]


def explore_nsl_kdd(base):
    """Examine NSL-KDD: file listing, label distribution, and a bar chart."""
    nsl_dir = os.path.join(base, "data", "raw", "nsl_kdd")
    all_files = os.listdir(nsl_dir)
    train_files = [f for f in all_files
                   if "train" in f.lower() and (f.endswith(".csv") or f.endswith(".txt"))]
    test_files  = [f for f in all_files
                   if "test" in f.lower() and (f.endswith(".csv") or f.endswith(".txt"))]

    print(f"\nNSL-KDD DATASET")
    print(f"  Files: {all_files}")
    print(f"  Train candidates: {train_files}")
    print(f"  Test candidates:  {test_files}")

    if not train_files:
        print("  No train file found.")
        return train_files, test_files

    filepath = os.path.join(nsl_dir, train_files[0])
    try:
        df = pd.read_csv(filepath, nrows=5000)
    except Exception:
        df = pd.read_csv(filepath, names=NSL_COLUMNS, nrows=5000)

    print(f"  Shape (sample): {df.shape}")
    for i, col in enumerate(df.columns, 1):
        print(f"    {i:2d}. {col}")

    if "label" in df.columns:
        print(f"\n  Label distribution (sample):")
        print(df["label"].value_counts().to_string())

        fig, ax = plt.subplots(figsize=(12, 5))
        df["label"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("NSL-KDD - Attack Type Distribution (sample)")
        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out = os.path.join(base, "results", "figures", "nsl_kdd_label_distribution.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Chart saved: {out}")

        df["is_attack"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)
        print(f"\n  Binary split (sample):")
        print(df["is_attack"].value_counts().to_string())

    return train_files, test_files


# ---- Main ------------------------------------------------------------------

def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)
    os.makedirs(os.path.join(base, "results", "figures"), exist_ok=True)

    print("DATA EXPLORATION\n")

    ds2os_file, ds2os_targets = explore_ds2os(base)
    nsl_train, nsl_test = explore_nsl_kdd(base)

    summary = {
        "ds2os_file": ds2os_file or "NOT_FOUND",
        "ds2os_targets": ds2os_targets if ds2os_targets else "MANUALLY_IDENTIFY",
        "nsl_train_file": nsl_train[0] if nsl_train else "NOT_FOUND",
        "nsl_test_file": nsl_test[0] if nsl_test else "NOT_FOUND",
    }
    out_path = os.path.join(base, "results", "exploration_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {out_path}")
    print("EXPLORATION COMPLETE")


if __name__ == "__main__":
    main()
