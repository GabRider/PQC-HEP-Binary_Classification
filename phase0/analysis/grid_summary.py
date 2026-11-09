"""
grid_summary.py - summary tables of the alpha x b grid (circuit 6, 100,000 events).

Reads results.json and history.csv in every run folder of phase0/results/grid/ and writes
two tables in phase0/figures/grid/:
  grid_runs.csv     one line per run (72 lines)
  grid_configs.csv  one line per (alpha, b): mean, min and max over the theta seeds (24 lines),
                    sorted by mean validation loss

Usage (from anywhere):  python phase0/analysis/grid_summary.py
"""

import json
import csv
from pathlib import Path
import numpy as np

# --- Paths ---------------------------------------------------------------------------
HERE = Path(__file__).parent                          # phase0/analysis/
RUNS = HERE.parent / "results" / "grid"               # the 72 run folders
OUT  = HERE.parent / "figures" / "grid"               # tables and figures of the grid
OUT.mkdir(parents=True, exist_ok=True)

# Circuit evaluations per event and per step for circuit 6: 1 at the current theta,
# 2 for each of the 16 single-qubit rotations, 4 for each of the 12 CRX gates.
N_SHIFTED = 81

# --- One line per run ----------------------------------------------------------------
rows = []
for folder in sorted(RUNS.glob("grid_hybrid_*")):
    info = json.loads((folder / "results.json").read_text())
    run  = info["run_info"]
    best = info["output"]["best_epoch"]
    with open(folder / "history.csv") as f:
        hist = list(csv.DictReader(f))
    val_loss = np.array([float(h["val_loss"]) for h in hist])

    n_events = run["n_events_used"]
    batch    = run["BATCH_SIZE"]
    alpha    = run["ALPHA"]
    n_epochs = run["N_ITER"]
    steps_ep = n_events // batch
    e_best   = best["epoch"]

    rows.append({
        "alpha"                 : alpha,
        "batch"                 : batch,
        "seed"                  : run["THETA_INIT_SEED"],
        "steps_per_epoch"       : steps_ep,
        "best_epoch"            : e_best,
        "steps_to_best"         : e_best * steps_ep,
        "circuit_evals_to_best" : N_SHIFTED * n_events * e_best,
        "budget_rad"            : alpha * n_epochs * steps_ep,
        "val_loss_best"         : best["val_loss"],
        "val_auc_best"          : best["val_auc"],
        "train_loss_best"       : float(hist[e_best]["train_loss"]),
        "val_loss_change_last10": val_loss[-1] - val_loss[-11],
    })

keys = {(r["alpha"], r["batch"], r["seed"]) for r in rows}
if len(rows) != 72 or len(keys) != 72:
    print(f"Warning: {len(rows)} runs found, {len(keys)} different (alpha, b, seed). Expected 72.")

with open(OUT / "grid_runs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# --- One line per (alpha, b) ---------------------------------------------------------
configs = []
for alpha, batch in sorted({(r["alpha"], r["batch"]) for r in rows}):
    group = [r for r in rows if r["alpha"] == alpha and r["batch"] == batch]
    loss  = np.array([r["val_loss_best"] for r in group])
    auc   = np.array([r["val_auc_best"]  for r in group])
    configs.append({
        "alpha"             : alpha,
        "batch"             : batch,
        "n_seeds"           : len(group),
        "val_loss_mean"     : loss.mean(),
        "val_loss_min"      : loss.min(),
        "val_loss_max"      : loss.max(),
        "val_auc_mean"      : auc.mean(),
        "best_epochs"       : " ".join(str(r["best_epoch"]) for r in group),
        "steps_to_best_mean": np.mean([r["steps_to_best"] for r in group]),
        "evals_to_best_mean": np.mean([r["circuit_evals_to_best"] for r in group]),
        "budget_rad"        : group[0]["budget_rad"],
    })
configs.sort(key=lambda c: c["val_loss_mean"])

with open(OUT / "grid_configs.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=configs[0].keys())
    writer.writeheader()
    writer.writerows(configs)

print(f"{len(rows)} runs, {len(configs)} configurations. Best 8 (mean validation loss over the seeds):")
for c in configs[:8]:
    print(f"  alpha = {c['alpha']:<6} b = {c['batch']:>5} : val loss {c['val_loss_mean']:.4f} "
          f"[{c['val_loss_min']:.4f}, {c['val_loss_max']:.4f}], AUC {c['val_auc_mean']:.4f}, "
          f"best epochs {c['best_epochs']}")
print(f"Tables saved in {OUT}/")