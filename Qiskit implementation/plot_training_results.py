
"""
Automatic diagnostic plots for PQC_phase0.py.

Generates:
    1. Weighted ROC curves: training vs validation
    2. Overtraining check: signal/background score distributions
    3. Training and validation loss vs epoch

Required input files:
    history.csv
    train_predictions.csv
    val_predictions.csv

Outputs:
    plots/ROC_train_vs_val.png
    plots/overtraining_check.png
    plots/loss_train_vs_val.png

Each figure is also saved as PDF.
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


# ============================================================
# DATA LOADING
# ============================================================

def _read_predictions(path):
    """Read labels, weights and classifier probabilities."""

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty predictions file: {path}")

    labels = np.array(
        [int(row["label"]) for row in rows], dtype=int
    )

    weights = np.array(
        [float(row["weight"]) for row in rows], dtype=float
    )

    probabilities = np.array(
        [float(row["p"]) for row in rows], dtype=float
    )

    if not np.all(np.isfinite(weights)):
        raise ValueError(f"Non-finite weights in {path}")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"Non-finite predictions in {path}")

    if np.any(weights < 0):
        raise ValueError(f"Negative weights in {path}")

    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise ValueError(f"Predictions outside [0, 1] in {path}")

    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(
            f"Both signal and background must be present in {path}"
        )

    for label in (0, 1):
        if weights[labels == label].sum() <= 0:
            raise ValueError(
                f"Nonpositive total weight for class {label} in {path}"
            )

    return labels, weights, probabilities


def _read_history(path):
    """Read per-epoch training and validation metrics."""

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"Empty history file: {path}")

    epochs = np.array(
        [int(row["epoch"]) for row in rows]
    )

    train_loss = np.array(
        [float(row["train_loss"]) for row in rows]
    )

    val_loss = np.array(
        [float(row["val_loss"]) for row in rows]
    )

    if not np.all(np.isfinite(train_loss)):
        raise ValueError("Non-finite training losses")

    if not np.all(np.isfinite(val_loss)):
        raise ValueError("Non-finite validation losses")

    return epochs, train_loss, val_loss


# ============================================================
# FIGURE SAVING
# ============================================================

def _save(fig, plots_dir, filename):
    """Save a figure in PNG and PDF formats."""

    fig.tight_layout()

    fig.savefig(
        plots_dir / f"{filename}.png",
        dpi=200,
        bbox_inches="tight"
    )

    fig.savefig(
        plots_dir / f"{filename}.pdf",
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# 1. WEIGHTED ROC CURVES
# ============================================================

def plot_roc_curves(
    train_labels,
    train_weights,
    train_probabilities,
    val_labels,
    val_weights,
    val_probabilities,
    plots_dir
):
    """Plot weighted training and validation ROC curves."""

    train_fpr, train_tpr, _ = roc_curve(
        train_labels,
        train_probabilities,
        sample_weight=train_weights
    )

    val_fpr, val_tpr, _ = roc_curve(
        val_labels,
        val_probabilities,
        sample_weight=val_weights
    )

    train_auc = roc_auc_score(
        train_labels,
        train_probabilities,
        sample_weight=train_weights
    )

    val_auc = roc_auc_score(
        val_labels,
        val_probabilities,
        sample_weight=val_weights
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    ax.plot(
        train_fpr,
        train_tpr,
        color="tab:blue",
        linewidth=2,
        label=f"Training (AUC = {train_auc:.4f})"
    )

    ax.plot(
        val_fpr,
        val_tpr,
        color="tab:orange",
        linewidth=2,
        label=f"Validation (AUC = {val_auc:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Random classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Weighted ROC Curves: Training vs Validation")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)

    _save(fig, plots_dir, "ROC_train_vs_val")

    print(
        f"  Weighted AUC: "
        f"train={train_auc:.6f}, "
        f"validation={val_auc:.6f}"
    )


# ============================================================
# 2. OVERTRAINING CHECK
# ============================================================

def plot_overtraining(
    train_labels,
    train_weights,
    train_probabilities,
    val_labels,
    val_weights,
    val_probabilities,
    plots_dir
):
    """
    Compare classifier-output distributions.

    Signal: blue
    Background: orange

    Validation: filled histograms
    Training: step histograms

    Each class and dataset is independently normalised
    to unit total weighted area.
    """

    fig, ax = plt.subplots(figsize=(8, 5.5))

    bins = np.linspace(0, 1, 31)

    datasets = [
        (
            "Training",
            train_labels,
            train_weights,
            train_probabilities,
            False
        ),
        (
            "Validation",
            val_labels,
            val_weights,
            val_probabilities,
            True
        )
    ]

    classes = [
        (1, "Signal", "tab:blue"),
        (0, "Background", "tab:orange")
    ]

    for dataset_name, labels, weights, probs, is_val in datasets:

        for class_label, class_name, color in classes:

            mask = labels == class_label

            total_weight = np.sum(weights[mask])

            if total_weight <= 0:
                raise ValueError(
                    f"Nonpositive total weight for "
                    f"{dataset_name} {class_name}"
                )

            normalized_weights = (
                weights[mask] / total_weight
            )

            ax.hist(
                probs[mask],
                bins=bins,
                weights=normalized_weights,
                histtype="stepfilled" if is_val else "step",
                alpha=0.25 if is_val else 1.0,
                linewidth=0 if is_val else 1.8,
                color=color,
                label=f"{class_name} - {dataset_name}"
            )

    ax.set_xlabel("Classifier Output P(signal)")
    ax.set_ylabel("Fraction of Weighted Events per Bin")

    ax.set_title(
        "Overtraining Check: Training vs Validation"
    )

    ax.set_xlim(0, 1)

    ax.legend()
    ax.grid(alpha=0.2)

    _save(fig, plots_dir, "overtraining_check")


# ============================================================
# 3. TRAINING AND VALIDATION LOSS
# ============================================================

def plot_losses(
    epochs,
    train_loss,
    val_loss,
    plots_dir
):
    """Plot training and validation loss versus epoch."""

    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    ax.plot(
        epochs,
        train_loss,
        color="tab:blue",
        linewidth=2,
        label="Training Loss"
    )

    ax.plot(
        epochs,
        val_loss,
        color="tab:orange",
        linewidth=2,
        label="Validation Loss"
    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel(
        "Weighted Binary Cross-Entropy / Number of Events"
    )

    ax.set_title(
        "Training and Validation Loss"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    _save(fig, plots_dir, "loss_train_vs_val")


# ============================================================
# MAIN FUNCTION
# ============================================================

def generate_all_plots(output_dir):
    """
    Generate all diagnostic plots from a training output folder.

    This function is called automatically by PQC_phase0.py
    when GENERATE_PLOTS=True.
    """

    output_dir = Path(output_dir)

    train_path = output_dir / "train_predictions.csv"
    val_path = output_dir / "val_predictions.csv"
    history_path = output_dir / "history.csv"

    for path in (train_path, val_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Read final predictions

    train_labels, train_weights, train_probs = (
        _read_predictions(train_path)
    )

    val_labels, val_weights, val_probs = (
        _read_predictions(val_path)
    )

    # Read learning curves

    epochs, train_loss, val_loss = (
        _read_history(history_path)
    )

    # Generate ROC curves

    plot_roc_curves(
        train_labels,
        train_weights,
        train_probs,
        val_labels,
        val_weights,
        val_probs,
        plots_dir
    )

    # Generate overtraining histogram

    plot_overtraining(
        train_labels,
        train_weights,
        train_probs,
        val_labels,
        val_weights,
        val_probs,
        plots_dir
    )

    # Generate loss curves

    plot_losses(
        epochs,
        train_loss,
        val_loss,
        plots_dir
    )

    print("  All diagnostic plots generated successfully.")

    return plots_dir


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Generate diagnostic plots for PQC training"
    )

    parser.add_argument(
        "output_dir",
        help="Path to a training output directory"
    )

    args = parser.parse_args()

    plots_dir = generate_all_plots(args.output_dir)

    print(f"Figures saved to: {plots_dir}")
