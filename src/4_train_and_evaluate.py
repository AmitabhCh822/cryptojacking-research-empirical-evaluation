"""
Model Training and Evaluation
Author: Amitabh Chakravorty

Trains six classical ML classifiers on both preprocessed datasets, collects
accuracy / F1 / precision / recall / timing metrics, plots confusion matrices
and comparison charts, and saves everything under results/.

Usage:
    python src/4_train_and_evaluate.py --base_path ./
"""

import os
import time
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument("--base_path", type=str, default=".",
                        help="Project root (default: current directory)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models():
    """Return an ordered dict of model name -> untrained estimator."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_split=5,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, n_jobs=-1, verbose=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, min_samples_split=5, random_state=42,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


# ---------------------------------------------------------------------------
# Train + evaluate one model
# ---------------------------------------------------------------------------

def run_model(model, name, X_tr, X_te, y_tr, y_te, dataset, models_dir):
    """Fit, predict, measure, save, and return a results dict + confusion matrix."""
    print(f"\n  {name} on {dataset.upper()}")

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_sec = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    infer_sec = time.time() - t0

    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_te, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"    Accuracy  {acc:.4f}  |  F1  {f1:.4f}  |  Train  {train_sec:.2f}s")

    # Save trained model
    safe_name = name.replace(" ", "_")
    pkl_path = os.path.join(models_dir, f"{safe_name}_{dataset}.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(model, fh)

    row = {
        "model": name, "dataset": dataset,
        "accuracy": acc, "f1_score": f1,
        "precision": prec, "recall": rec,
        "train_time_sec": train_sec, "inference_time_sec": infer_sec,
        "samples_train": len(y_tr), "samples_test": len(y_te),
        "features": X_tr.shape[1],
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
    }
    return row, cm


# ---------------------------------------------------------------------------
# Run all models on one dataset
# ---------------------------------------------------------------------------

def train_on_dataset(dataset, base):
    """Load processed arrays, train every model, return results list + CMs."""
    proc = os.path.join(base, "data", "processed")
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)

    X_tr = np.load(os.path.join(proc, f"X_train_{dataset}.npy"))
    X_te = np.load(os.path.join(proc, f"X_test_{dataset}.npy"))
    y_tr = np.load(os.path.join(proc, f"y_train_{dataset}.npy"))
    y_te = np.load(os.path.join(proc, f"y_test_{dataset}.npy"))

    print(f"\n{'=' * 50}")
    print(f"  DATASET: {dataset.upper()}")
    print(f"  Train {X_tr.shape[0]:,} x {X_tr.shape[1]}   Test {X_te.shape[0]:,}")
    print(f"{'=' * 50}")

    results, cms = [], []
    for name, model in build_models().items():
        try:
            row, cm = run_model(model, name, X_tr, X_te, y_tr, y_te,
                                dataset, models_dir)
            results.append(row)
            cms.append((name, cm))
        except Exception as exc:
            print(f"    ERROR training {name}: {exc}")

    return results, cms


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_confusion_grids(cms, dataset, fig_dir):
    """2x3 grid of confusion matrices."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    axes = axes.ravel()

    for idx, (name, cm) in enumerate(cms):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    annot_kws={"size": 16, "fontweight": "bold"})
        axes[idx].set_title(name, fontsize=18, fontweight="bold", pad=12)
        axes[idx].set_ylabel("True Label", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label", fontsize=14, fontweight="bold")
        axes[idx].set_xticklabels(["Normal", "Attack"], fontsize=13)
        axes[idx].set_yticklabels(["Normal", "Attack"], fontsize=13, rotation=0)

    for idx in range(len(cms), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Confusion Matrices - {dataset.upper()}",
                 fontsize=22, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(fig_dir, f"confusion_matrices_{dataset}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_accuracy_bars(df, fig_dir):
    """Grouped bar chart comparing accuracy across datasets."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    datasets = df["dataset"].unique()
    models = sorted(df["model"].unique())
    x = np.arange(len(models))
    w = 0.35

    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].set_index("model").loc[models]
        bars = ax.bar(x + i * w, sub["accuracy"], w, label=ds.upper(), alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xlabel("Model", fontsize=18, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=18, fontweight="bold")
    ax.set_title("Model Accuracy Comparison Across Datasets",
                 fontsize=22, fontweight="bold", pad=15)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim([0.74, 1.01])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_f1_bars(df, fig_dir):
    """Grouped bar chart comparing F1 across datasets."""
    fig, ax = plt.subplots(figsize=(14, 7))

    datasets = df["dataset"].unique()
    models = sorted(df["model"].unique())
    x = np.arange(len(models))
    w = 0.35

    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].set_index("model").loc[models]
        bars = ax.bar(x + i * w, sub["f1_score"], w, label=ds.upper(), alpha=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=14, fontweight="bold")

    ax.set_xlabel("Model", fontsize=18, fontweight="bold")
    ax.set_ylabel("F1-Score", fontsize=18, fontweight="bold")
    ax.set_title("Model F1-Score Comparison Across Datasets",
                 fontsize=22, fontweight="bold", pad=15)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16)
    ax.set_ylim([0.74, 1.01])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "f1_comparison.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_cost_tradeoff(df, fig_dir):
    """Scatter: training time vs accuracy."""
    fig, ax = plt.subplots(figsize=(16, 8))

    palette = {"ds2os": "#3182CE", "nsl_kdd": "#38A169"}
    markers = {"ds2os": "o", "nsl_kdd": "s"}

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        ax.scatter(sub["train_time_sec"], sub["accuracy"],
                   s=220, alpha=0.85, label=ds.upper(),
                   color=palette.get(ds, "#999"), marker=markers.get(ds, "o"),
                   edgecolors="white", linewidth=1.5, zorder=5)

        for _, row in sub.iterrows():
            ax.annotate(row["model"],
                        xy=(row["train_time_sec"], row["accuracy"]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=12, fontweight="bold",
                        color=palette.get(ds, "#999"))

    ax.set_xlabel("Training Time (seconds)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=18, fontweight="bold")
    ax.set_title("Computational Cost vs Performance",
                 fontsize=22, fontweight="bold", pad=15)
    ax.legend(fontsize=16, markerscale=1.4)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(fig_dir, "computational_cost.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_heatmap(df, fig_dir):
    """Accuracy heatmap (models x datasets)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    pivot = df.pivot_table(values="accuracy", index="model", columns="dataset")
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="RdYlGn",
                vmin=0.75, vmax=1.0, linewidths=0.5, ax=ax,
                annot_kws={"size": 18, "fontweight": "bold"},
                cbar_kws={"label": "Accuracy"})
    ax.set_title("Accuracy Heatmap", fontsize=22, fontweight="bold", pad=15)
    ax.set_xlabel("Dataset", fontsize=18, fontweight="bold")
    ax.set_ylabel("Model", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    plt.tight_layout()

    path = os.path.join(fig_dir, "accuracy_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(df):
    print(f"\n{'=' * 50}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 50}")

    cols = ["model", "dataset", "accuracy", "f1_score", "precision", "recall", "train_time_sec"]
    print(df[cols].to_string(index=False))

    best = df.loc[df["accuracy"].idxmax()]
    print(f"\nHighest accuracy: {best['model']} on {best['dataset'].upper()} "
          f"- {best['accuracy']:.4f}")

    print(f"Mean accuracy:  {df['accuracy'].mean():.4f}")
    print(f"Mean F1-score:  {df['f1_score'].mean():.4f}")
    print(f"Total training: {df['train_time_sec'].sum():.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    base = os.path.abspath(args.base_path)
    fig_dir = os.path.join(base, "results", "figures")
    met_dir = os.path.join(base, "results", "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    print("MODEL TRAINING AND EVALUATION\n")

    all_rows = []
    for dataset in ("ds2os", "nsl_kdd"):
        rows, cms = train_on_dataset(dataset, base)
        all_rows.extend(rows)
        if cms:
            plot_confusion_grids(cms, dataset, fig_dir)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(met_dir, "model_performance.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Metrics saved to {csv_path}")

    # Generate comparison plots
    print("\nGenerating visualizations ...")
    plot_accuracy_bars(df, fig_dir)
    plot_f1_bars(df, fig_dir)
    plot_cost_tradeoff(df, fig_dir)
    plot_heatmap(df, fig_dir)

    print_summary(df)
    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()
