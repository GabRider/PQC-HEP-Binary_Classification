"""
plot_phase0.py - figures of Phase 0 (learning-rate scan, batch-size scan, standardisation).

Reads the history.csv and val_predictions.csv files written by PQC_phase0.py in the run
folders of phase0/results/, and saves the figures (PDF for LaTeX, PNG for slides and
e-mails) in phase0/figures/.

Usage (from anywhere):  python phase0/analysis/plot_phase0.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# --- Paths ---------------------------------------------------------------------------
HERE = Path(__file__).parent                          # phase0/analysis/
RUNS = HERE.parent / "results"                        # folder containing the run folders
OUT  = HERE.parent / "figures"                        # output folder for the figures
OUT.mkdir(exist_ok=True)

# --- Colours (one per learning rate / batch size) ------------------------------------
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GREY = "#7a7a7a"                                       # runs without standardisation


# --- Reading a run -------------------------------------------------------------------
def find_run(prefix):
    """Return the run folder whose name starts with `prefix` (followed by the timestamp)."""
    found = sorted(RUNS.glob(prefix + "_2026*"))
    if len(found) != 1:
        raise FileNotFoundError(f"{len(found)} folder(s) found for {prefix}")
    return found[0]

def read_history(prefix):
    """Read history.csv (columns: epoch, step, train_loss, train_auc, val_loss, val_auc)."""
    return np.genfromtxt(find_run(prefix) / "history.csv", delimiter=",", names=True)

def read_predictions(prefix):
    """Read val_predictions.csv (columns: label, weight, p)."""
    return np.genfromtxt(find_run(prefix) / "val_predictions.csv", delimiter=",", names=True)

def save(fig, name):
    """Save the figure as PDF (vector, for the LaTeX report) and PNG (for slides and e-mails)."""
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  saved: {name}")


# --- Phase 0 runs --------------------------------------------------------------------
ALPHAS = ["0.001", "0.005", "0.01", "0.05", "0.1"]
scan_alpha = {a: read_history(f"circuit6_layers1_50iter_batch100_alpha{a}") for a in ALPHAS}

BATCHES = ["10", "100", "1000"]
scan_batch = {b: read_history(f"circuit6_layers1_50iter_batch{b}_alpha0.05") for b in BATCHES}

ALPHAS_RAW = ["0.005", "0.05"]
raw = {a: read_history(f"raw_circuit6_layers1_50iter_batch100_alpha{a}") for a in ALPHAS_RAW}

for a, h in scan_alpha.items():
    print(f"alpha = {a:>5}: {len(h)} rows, final validation AUC = {h['val_auc'][-1]:.3f}")


# =====================================================================================
# FIGURE 1 - Learning-rate scan: validation AUC per epoch
# =====================================================================================
fig, ax = plt.subplots(figsize=(6, 4))
for colour, (a, h) in zip(COLOURS, scan_alpha.items()):
    ax.plot(h["epoch"], h["val_auc"], color=colour, linewidth=2, label=f"α = {a}")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC (validation)")
ax.set_title("Learning-rate scan (circuit 6, L = 1, b = 100)")
ax.grid(alpha=0.3)
ax.legend(title="Learning rate")
save(fig, "fig1_scan_alpha_auc")


# =====================================================================================
# FIGURE 2 - Learning-rate scan: training and validation loss, one panel per alpha
# =====================================================================================
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), sharey=True)
for ax, colour, (a, h) in zip(axes, COLOURS, scan_alpha.items()):
    ax.plot(h["epoch"], h["train_loss"], color=colour, linestyle="--", linewidth=1.5, label="train")
    ax.plot(h["epoch"], h["val_loss"],   color=colour, linewidth=2,                  label="validation")
    i_min = np.argmin(h["val_loss"])                   # dot = minimum of the validation loss
    ax.plot(h["epoch"][i_min], h["val_loss"][i_min], "o", color=colour, markersize=8)
    ax.set_title(f"α = {a}")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Loss (weighted cross-entropy)")
fig.suptitle("Training and validation loss vs. learning rate (circuit 6, L = 1, b = 100)")
save(fig, "fig2_scan_alpha_loss")


# =====================================================================================
# FIGURE 3 - Batch-size scan: validation AUC per epoch (equal computing cost)
# =====================================================================================
fig, ax = plt.subplots(figsize=(6, 4))
for colour, (b, h) in zip(COLOURS, scan_batch.items()):
    ax.plot(h["epoch"], h["val_auc"], color=colour, linewidth=2, label=f"b = {b} ({int(b) // 10} %)")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC (validation)")
ax.set_title("Batch-size scan (circuit 6, L = 1, α = 0.05)")
ax.grid(alpha=0.3)
ax.legend(title="Batch size")
save(fig, "fig3_scan_batch_auc")


# =====================================================================================
# FIGURE 4 - Standardisation: validation AUC, z-score (solid) vs. raw features (dashed)
# =====================================================================================
fig, ax = plt.subplots(figsize=(6, 4))
for a in ALPHAS_RAW:
    colour = COLOURS[ALPHAS.index(a)]                  # same colour as in figure 1
    ax.plot(scan_alpha[a]["epoch"], scan_alpha[a]["val_auc"], color=colour, linewidth=2,
            label=f"z-score, α = {a}")
    ax.plot(raw[a]["epoch"], raw[a]["val_auc"], color=colour, linewidth=2, linestyle="--",
            label=f"raw, α = {a}")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC (validation)")
ax.set_title("Effect of standardisation (circuit 6, L = 1, b = 100)")
ax.grid(alpha=0.3)
ax.legend()
save(fig, "fig4_normalisation_auc")


# =====================================================================================
# FIGURE 5 - ROC curve on validation (alpha = 0.05, b = 100, end of training)
# =====================================================================================
fig, ax = plt.subplots(figsize=(5, 5))
colour = COLOURS[ALPHAS.index("0.05")]
roc_runs = [
    ("z-score",      "circuit6_layers1_50iter_batch100_alpha0.05",     "-"),
    ("raw features", "raw_circuit6_layers1_50iter_batch100_alpha0.05", "--"),
]
for name, prefix, style in roc_runs:
    d = read_predictions(prefix)
    fpr, tpr, _ = roc_curve(d["label"], d["p"], sample_weight=d["weight"])
    auc = roc_auc_score(d["label"], d["p"], sample_weight=d["weight"])
    ax.plot(fpr, tpr, color=colour, linestyle=style, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1, label="random (AUC = 0.5)")
ax.set_xlabel("False positive rate (background efficiency)")
ax.set_ylabel("True positive rate (signal efficiency)")
ax.set_title("ROC curve on validation (α = 0.05, b = 100, epoch 50)")
ax.set_aspect("equal")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
save(fig, "fig5_roc")


# =====================================================================================
# FIGURE 6 - Batch-size scan: training and validation loss, one panel per b
# =====================================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, colour, (b, h) in zip(axes, COLOURS, scan_batch.items()):
    ax.plot(h["epoch"], h["train_loss"], color=colour, linestyle="--", linewidth=1.5, label="train")
    ax.plot(h["epoch"], h["val_loss"],   color=colour, linewidth=2,                  label="validation")
    i_min = np.argmin(h["val_loss"])
    ax.plot(h["epoch"][i_min], h["val_loss"][i_min], "o", color=colour, markersize=8)
    ax.set_title(f"b = {b} ({int(b) // 10} %)")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Loss (weighted cross-entropy)")
fig.suptitle("Training and validation loss vs. batch size (circuit 6, L = 1, α = 0.05)")
save(fig, "fig6_scan_batch_loss")


# =====================================================================================
# FIGURE 7 - Standardisation: training and validation loss, z-score (colour) vs. raw (grey)
# =====================================================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, a in zip(axes, ALPHAS_RAW):
    colour = COLOURS[ALPHAS.index(a)]
    z, r = scan_alpha[a], raw[a]
    ax.plot(z["epoch"], z["train_loss"], color=colour, linestyle="--", linewidth=1.5, label="z-score, train")
    ax.plot(z["epoch"], z["val_loss"],   color=colour, linewidth=2,                  label="z-score, validation")
    ax.plot(r["epoch"], r["train_loss"], color=GREY,   linestyle="--", linewidth=1.5, label="raw, train")
    ax.plot(r["epoch"], r["val_loss"],   color=GREY,   linewidth=2,                  label="raw, validation")
    ax.set_title(f"α = {a}")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Loss (weighted cross-entropy)")
fig.suptitle("Training and validation loss: z-score vs. raw features (circuit 6, L = 1, b = 100)")
save(fig, "fig7_normalisation_loss")



# --- Runs with 10,000 training events (batch-size scan, alpha = 0.05) ----------------
BATCHES_10K = ["100", "1000", "2000"]
scan_10k = {b: read_history(f"10k_circuit6_layers1_50iter_batch{b}_alpha0.05") for b in BATCHES_10K}

for b, h in scan_10k.items():
    print(f"10k, b = {b:>4}: {len(h)} rows, final validation AUC = {h['val_auc'][-1]:.3f}")


# =====================================================================================
# FIGURE 8 - Batch-size scan with 10,000 events: validation AUC per epoch
# =====================================================================================
fig, ax = plt.subplots(figsize=(6, 4))
for colour, (b, h) in zip(COLOURS, scan_10k.items()):
    ax.plot(h["epoch"], h["val_auc"], color=colour, linewidth=2, label=f"b = {b} ({int(b) // 100} %)")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC (validation)")
ax.set_title("Batch-size scan, 10,000 events (circuit 6, L = 1, α = 0.05)")
ax.grid(alpha=0.3)
ax.legend(title="Batch size")
save(fig, "fig8_10k_batch_auc")


# =====================================================================================
# FIGURE 9 - Batch-size scan with 10,000 events: training and validation loss
# =====================================================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, colour, (b, h) in zip(axes, COLOURS, scan_10k.items()):
    ax.plot(h["epoch"], h["train_loss"], color=colour, linestyle="--", linewidth=1.5, label="train")
    ax.plot(h["epoch"], h["val_loss"],   color=colour, linewidth=2,                  label="validation")
    i_min = np.argmin(h["val_loss"])
    ax.plot(h["epoch"][i_min], h["val_loss"][i_min], "o", color=colour, markersize=8)
    ax.set_title(f"b = {b} ({int(b) // 100} %)")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Loss (weighted cross-entropy)")
fig.suptitle("Training and validation loss vs. batch size, 10,000 events (circuit 6, L = 1, α = 0.05)")
save(fig, "fig9_10k_batch_loss")


# =====================================================================================
# FIGURE 10 - Effect of the number of events: 1,000 vs. 10,000 (batch = 10 %, 500 steps)
# =====================================================================================
fig, ax = plt.subplots(figsize=(6, 4))
runs_10pct = [
    ("1,000 events (b = 100)",   scan_batch["100"], COLOURS[0]),
    ("10,000 events (b = 1000)", scan_10k["1000"],  COLOURS[1]),
]
for name, h, colour in runs_10pct:
    ax.plot(h["epoch"], h["train_auc"], color=colour, linestyle="--", linewidth=1.5, label=f"{name}, train")
    ax.plot(h["epoch"], h["val_auc"],   color=colour, linewidth=2,                  label=f"{name}, validation")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("1,000 vs. 10,000 events (batch = 10 %, α = 0.05)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
save(fig, "fig10_1k_vs_10k")