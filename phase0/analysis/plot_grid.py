"""
plot_grid.py - figures of the alpha x b grid (circuit 6, 100,000 events, 3 theta seeds).

Reads grid_configs.csv (written by grid_summary.py) and the history.csv files of the run
folders in phase0/results/grid/, and saves the figures (PDF for LaTeX, PNG for slides and
e-mails) in phase0/figures/grid/.

Usage (from anywhere):  python phase0/analysis/plot_grid.py
Run grid_summary.py first.
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Paths ---------------------------------------------------------------------------
HERE = Path(__file__).parent                          # phase0/analysis/
RUNS = HERE.parent / "results" / "grid"               # the 72 run folders
OUT  = HERE.parent / "figures" / "grid"               # tables (input) and figures (output)

# --- Grid values and colours ---------------------------------------------------------
ALPHAS  = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
BATCHES = [1000, 10000, 20000, 50000]
SEEDS   = [1, 2, 3]

# One colour per learning rate (same first five colours as plot_phase0.py).
ALPHA_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# One shade of blue per batch size (light = small b, dark = large b) and one marker shape.
BATCH_COLOURS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BATCH_MARKERS = ["o", "s", "^", "D"]
GRID_COLOUR = "#e4e3df"
INK   = "#0b0b0b"
MUTED = "#52514e"

plt.rcParams.update({"axes.edgecolor": "#9a9993", "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED})

def save(fig, name):
    """Save the figure as PDF (vector, for the LaTeX report) and PNG (for slides and e-mails)."""
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {name}")

def light_grid(ax):
    ax.grid(True, color=GRID_COLOUR, linewidth=0.6)
    ax.set_axisbelow(True)

def b_label(b):
    return f"b = {b:,}".replace(",", " ")

# --- Reading the data ----------------------------------------------------------------
with open(OUT / "grid_configs.csv") as f:
    configs = {(float(r["alpha"]), int(r["batch"])): r for r in csv.DictReader(f)}

def val(alpha, batch, column):
    """Value of one column of grid_configs.csv for the configuration (alpha, batch)."""
    return float(configs[(alpha, batch)][column])

def read_val_loss(alpha, batch, seed):
    """Validation loss at each epoch (0 to 50) of one run, read from its history.csv."""
    folder = next(RUNS.glob(f"grid_hybrid_*_batch{batch}_alpha{alpha}_seed{seed}_*"))
    with open(folder / "history.csv") as f:
        return np.array([float(h["val_loss"]) for h in csv.DictReader(f)])

# =====================================================================================
# G1 - heatmap of the mean validation loss
# =====================================================================================
VMIN, VMAX = 0.606, 0.630
blues = LinearSegmentedColormap.from_list(
    "blues", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"])

loss = np.array([[val(a, b, "val_loss_mean") for b in BATCHES] for a in ALPHAS])
lo   = np.array([[val(a, b, "val_loss_min")  for b in BATCHES] for a in ALPHAS])
hi   = np.array([[val(a, b, "val_loss_max")  for b in BATCHES] for a in ALPHAS])

fig, ax = plt.subplots(figsize=(6.5, 5.2))
im = ax.imshow(loss, cmap=blues, vmin=VMIN, vmax=VMAX, aspect="auto")
for i in range(len(ALPHAS)):
    for j in range(len(BATCHES)):
        colour = "white" if loss[i, j] < VMIN + 0.45 * (VMAX - VMIN) else INK
        ax.text(j, i - 0.12, f"{loss[i, j]:.4f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=colour)
        ax.text(j, i + 0.22, f"{lo[i, j]:.4f}–{hi[i, j]:.4f}", ha="center", va="center",
                fontsize=7, color=colour)
ax.set_xticks(range(len(BATCHES)), [f"{b:,}".replace(",", " ") for b in BATCHES])
ax.set_yticks(range(len(ALPHAS)), [str(a) for a in ALPHAS])
ax.set_xlabel("batch size b")
ax.set_ylabel("learning rate α")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cbar = fig.colorbar(im, ax=ax, extend="max")
cbar.set_label("validation loss at the best epoch (mean of 3 seeds)")
cbar.outline.set_visible(False)
ax.set_title("Validation loss, circuit 6, 100k events, 50 epochs", fontsize=11)
save(fig, "G1_heatmap_val_loss")

# =====================================================================================
# G2 - validation loss per epoch: one panel per batch size, one curve per alpha
# =====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True, sharey=True)
for ax, b in zip(axes.flat, BATCHES):
    for a, colour in zip(ALPHAS, ALPHA_COLOURS):
        curves = np.array([read_val_loss(a, b, s) for s in SEEDS])
        epochs = np.arange(curves.shape[1])
        ax.fill_between(epochs, curves.min(axis=0), curves.max(axis=0),
                        color=colour, alpha=0.18, linewidth=0)
        ax.plot(epochs, curves.mean(axis=0), color=colour, linewidth=2, label=f"α = {a}")
    ax.set_title(f"{b_label(b)}  ({100000 // b} steps per epoch)", fontsize=10)
    ax.set_ylim(0.60, 0.70)
    light_grid(ax)
for ax in axes[1, :]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("validation loss")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.tight_layout()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
           ncol=len(ALPHAS), fontsize=9, frameon=False)
save(fig, "G2_val_loss_vs_epoch")

# =====================================================================================
# G3 - best validation loss against alpha x number of steps
# =====================================================================================
fig, ax = plt.subplots(figsize=(6.5, 4.3))
for b, colour, marker in zip(BATCHES, BATCH_COLOURS, BATCH_MARKERS):
    x  = np.array([val(a, b, "budget_rad")    for a in ALPHAS])
    y  = np.array([val(a, b, "val_loss_mean") for a in ALPHAS])
    yl = y - np.array([val(a, b, "val_loss_min") for a in ALPHAS])
    yh = np.array([val(a, b, "val_loss_max") for a in ALPHAS]) - y
    ax.errorbar(x, y, yerr=[yl, yh], fmt=marker, color=colour, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, elinewidth=1, capsize=0,
                label=b_label(b))
ax.axvline(2, color="#9a9993", linewidth=1)
ax.text(2.2, 0.712, "2 rad", fontsize=8, color=MUTED)
ax.set_xscale("log")
ax.set_xlabel("α × number of steps in 50 epochs (rad)")
ax.set_ylabel("validation loss at the best epoch")
light_grid(ax)
ax.legend(fontsize=8, frameon=False)
save(fig, "G3_val_loss_vs_budget")

# =====================================================================================
# G4 - cost to reach the best epoch against validation loss (configurations near the best)
# =====================================================================================
LOSS_CUT = 0.612
points = []
for b, colour, marker in zip(BATCHES, BATCH_COLOURS, BATCH_MARKERS):
    for a in ALPHAS:
        y = val(a, b, "val_loss_mean")
        if y <= LOSS_CUT:
            epochs = [int(e) for e in configs[(a, b)]["best_epochs"].split()]
            converged = sum(e < 45 for e in epochs) >= 2      # at least 2 seeds stop before epoch 45
            points.append((val(a, b, "evals_to_best_mean"), y, a, b, colour, marker, converged))

fig, ax = plt.subplots(figsize=(7.5, 4.8))
right_side = True
for x, y, a, b, colour, marker, converged in sorted(points, key=lambda p: p[1]):
    ax.plot(x, y, marker, markersize=8, color=colour,
            markerfacecolor=colour if converged else "white", markeredgewidth=1.5)
    if x > 3.9e8:                                   # the pile of runs still improving at epoch 50
        right_side = not right_side
        offset, ha = ((7, 0), "left") if right_side else ((-7, 0), "right")
    else:
        offset, ha = (7, 0), "left"
    ax.annotate(f"α = {a}, {b_label(b)}", (x, y), xytext=offset, textcoords="offset points",
                fontsize=7, color=MUTED, ha=ha, va="center")
ax.set_xscale("log")
ax.set_xlim(1.5e8, 5.5e8)
ax.set_xlabel("circuit evaluations up to the best epoch (81 × 100 000 × best epoch)")
ax.set_ylabel("validation loss at the best epoch (mean of 3 seeds)")
light_grid(ax)
handles = [plt.Line2D([], [], linestyle="", marker=m, color=c, markersize=8, label=b_label(b))
           for b, c, m in zip(BATCHES, BATCH_COLOURS, BATCH_MARKERS)]
handles.append(plt.Line2D([], [], linestyle="", marker="o", color="#9a9993", markerfacecolor="white",
                          markeredgewidth=1.5, markersize=8,
                          label="best epoch ≥ 45 for at least 2 of the 3 seeds"))
ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
ax.set_title(f"Cost against performance, configurations with mean validation loss ≤ {LOSS_CUT}",
             fontsize=10)
save(fig, "G4_cost_vs_val_loss")