"""
Data Preprocessing
Author: Amitabh Chakravorty

Cleans both datasets, removes leakage-prone features from DS2OS, encodes
categoricals, applies SMOTE where needed, scales features, and saves the
processed arrays to data/processed/.

Usage:
    python src/3_preprocess.py --base_path ./
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess datasets")
    parser.add_argument("--base_path", type=str, default=".",
                        help="Project root (default: current directory)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DS2OS preprocessing
# ---------------------------------------------------------------------------

def preprocess_ds2os(base):
    """
    DS2OS pipeline:
      1. Load raw CSV
      2. Drop identifier / leakage columns
      3. Encode categoricals
      4. Binary target (normal = 0, attack = 1)
      5. Stratified 70/30 split (random_state=42)
      6. SMOTE on training set only (severe imbalance)
      7. StandardScaler
      8. Save .npy arrays and scaler/encoder pickles
    """
    print("\nPREPROCESSING DS2OS")

    filepath = os.path.join(base, "data", "raw", "ds2os", "DS2OS.csv")
    df = pd.read_csv(filepath)
    print(f"  Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Columns to drop: target + identifiers that leak label information
    drop_cols = [
        "normality",               # target
        "timestamp",
        "value",
        "sourceID",
        "sourceAddress",
        "accessedNodeAddress",
        "accessedNodeType",
        "operation",
    ]

    target_col = "normality"
    X = df.drop(drop_cols, axis=1)
    y = df[target_col]
    print(f"  Features retained: {X.shape[1]}")
    print(f"  Target distribution:\n{y.value_counts().to_string()}")

    # Encode categorical features
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    print(f"  Encoded {len(cat_cols)} categorical column(s)")

    # Fill any residual missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean())

    # Binary target
    y_bin = y.apply(lambda v: 0 if v == "normal" else 1)
    print(f"  Binary: Normal={int((y_bin == 0).sum()):,}  Attack={int((y_bin == 1).sum()):,}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_bin, test_size=0.3, random_state=42, stratify=y_bin
    )
    print(f"  Train: {X_train.shape[0]:,}   Test: {X_test.shape[0]:,}")

    # SMOTE for severe imbalance
    counts = np.bincount(y_train)
    ratio = counts[1] / counts[0] if counts[0] > 0 else 0
    if ratio < 0.3 or ratio > 3.0:
        print(f"  Imbalance ratio {ratio:.3f} -> applying SMOTE on training set")
        k = min(5, int((y_train == 1).sum()) - 1)
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE: Normal={int((y_train == 0).sum()):,}  Attack={int((y_train == 1).sum()):,}")

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save
    proc = os.path.join(base, "data", "processed")
    np.save(os.path.join(proc, "X_train_ds2os.npy"), X_train_sc)
    np.save(os.path.join(proc, "X_test_ds2os.npy"),  X_test_sc)
    np.save(os.path.join(proc, "y_train_ds2os.npy"), y_train.values if hasattr(y_train, "values") else y_train)
    np.save(os.path.join(proc, "y_test_ds2os.npy"),  y_test.values  if hasattr(y_test, "values")  else y_test)

    with open(os.path.join(proc, "scaler_ds2os.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(proc, "label_encoders_ds2os.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    print(f"  Saved to {proc}")
    print(f"  Final train shape: {X_train_sc.shape}   test shape: {X_test_sc.shape}")


# ---------------------------------------------------------------------------
# NSL-KDD preprocessing
# ---------------------------------------------------------------------------

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


def preprocess_nsl_kdd(base):
    """
    NSL-KDD pipeline:
      1. Load KDDTrain+.txt and KDDTest+.txt with standard column names
      2. Drop difficulty column
      3. Binary target (normal = 0, attack = 1)
      4. Label-encode categoricals (fit on union of train + test)
      5. StandardScaler
      6. Save arrays and pickles

    No SMOTE: KDDTrain+ is roughly 53/47 normal-to-attack.
    KDDTest+ is the official holdout containing attack types absent from training.
    """
    print("\nPREPROCESSING NSL-KDD")

    raw = os.path.join(base, "data", "raw", "nsl_kdd")
    df_train = pd.read_csv(os.path.join(raw, "KDDTrain+.txt"), names=NSL_COLUMNS)
    df_test  = pd.read_csv(os.path.join(raw, "KDDTest+.txt"),  names=NSL_COLUMNS)
    print(f"  Train: {df_train.shape[0]:,} rows   Test: {df_test.shape[0]:,} rows")

    # Drop difficulty score
    df_train = df_train.drop("difficulty", axis=1)
    df_test  = df_test.drop("difficulty",  axis=1)

    # Binary labels
    df_train["binary"] = df_train["label"].apply(lambda x: 0 if x == "normal" else 1)
    df_test["binary"]  = df_test["label"].apply(lambda x: 0 if x == "normal" else 1)
    print(f"  Train binary: Normal={int((df_train['binary']==0).sum()):,}  Attack={int((df_train['binary']==1).sum()):,}")
    print(f"  Test  binary: Normal={int((df_test['binary']==0).sum()):,}  Attack={int((df_test['binary']==1).sum()):,}")

    # Separate features / target
    X_train = df_train.drop(["label", "binary"], axis=1)
    X_test  = df_test.drop(["label", "binary"],  axis=1)
    y_train = df_train["binary"].values
    y_test  = df_test["binary"].values

    # Encode categoricals
    cat_cols = ["protocol_type", "service", "flag"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]])
        le.fit(combined)
        X_train[col] = le.transform(X_train[col])
        X_test[col]  = le.transform(X_test[col])
        encoders[col] = le
    print(f"  Encoded {len(cat_cols)} categorical column(s)")

    X_train = X_train.values
    X_test  = X_test.values

    # Scale
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save
    proc = os.path.join(base, "data", "processed")
    np.save(os.path.join(proc, "X_train_nsl_kdd.npy"), X_train_sc)
    np.save(os.path.join(proc, "X_test_nsl_kdd.npy"),  X_test_sc)
    np.save(os.path.join(proc, "y_train_nsl_kdd.npy"), y_train)
    np.save(os.path.join(proc, "y_test_nsl_kdd.npy"),  y_test)

    with open(os.path.join(proc, "scaler_nsl_kdd.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(proc, "label_encoders_nsl_kdd.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    print(f"  Saved to {proc}")
    print(f"  Final train shape: {X_train_sc.shape}   test shape: {X_test_sc.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)
    os.makedirs(os.path.join(base, "data", "processed"), exist_ok=True)

    print("DATA PREPROCESSING\n")

    preprocess_ds2os(base)
    preprocess_nsl_kdd(base)

    # Quick summary of what was saved
    proc = os.path.join(base, "data", "processed")
    saved = sorted(f for f in os.listdir(proc) if f.endswith((".npy", ".pkl")))
    print(f"\n{len(saved)} files written to {proc}:")
    for fname in saved:
        size_mb = os.path.getsize(os.path.join(proc, fname)) / (1024 ** 2)
        print(f"  {fname:<45s} {size_mb:>7.2f} MB")

    print("\nPREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
