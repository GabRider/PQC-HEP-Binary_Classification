"""
compare_hybrid.py

Compares each hybrid run in phase0/results with the old (qiskit) run that has the
same settings: history.csv, val_predictions.csv, final theta, and run time.
Read-only: no file is modified.

Usage (from the repository root):
    python phase0/analysis/compare_hybrid.py
"""
import json
from pathlib import Path
import numpy as np

HERE    = Path(__file__).parent          # phase0/analysis/
RESULTS = HERE.parent / "results"        # phase0/results/
LOGS    = HERE.parent / "logs"           # phase0/logs/


def read_settings(folder):
    """Return (settings, backend) of a run, read from its results.json."""
    info = json.loads((folder / "results.json").read_text())
    run = info["run_info"]
    settings = (
        info["circuit"]["circuit_number"],
        info["circuit"]["n_layers"],
        run["n_events_used"],
        run["N_ITER"],
        run["BATCH_SIZE"],
        run["ALPHA"],
        run["FEATURE_SCALING"],
    )
    backend = run.get("SIM_BACKEND", "qiskit")
    return settings, backend


def read_elapsed_times():
    """Return {run folder name: elapsed time in seconds}, read from the log files."""
    times = {}
    for log in LOGS.glob("*.txt"):
        folder, elapsed = None, None
        for line in log.read_text(errors="ignore").splitlines():
            if "Saving outputs to:" in line:
                folder = line.split("Saving outputs to:")[1].strip().rstrip("/")
            if "Elapsed time :" in line:
                elapsed = float(line.split(":")[1].split("s")[0])
        if folder is not None and elapsed is not None:
            times[folder] = elapsed
    return times





def max_diff_history(a, b):
    """Largest |difference| between two history.csv (all epochs, all columns)."""
    ha = np.genfromtxt(a / "history.csv", delimiter=",", names=True)
    hb = np.genfromtxt(b / "history.csv", delimiter=",", names=True)
    return max(np.max(np.abs(ha[col] - hb[col])) for col in ha.dtype.names)


def max_diff_predictions(a, b):
    """Largest |difference| between the validation p of two runs."""
    pa = np.genfromtxt(a / "val_predictions.csv", delimiter=",", names=True)["p"]
    pb = np.genfromtxt(b / "val_predictions.csv", delimiter=",", names=True)["p"]
    return np.max(np.abs(pa - pb))


def max_diff_theta(a, b):
    """Largest |difference| between the final theta of two runs."""
    ta = np.array(json.loads((a / "results.json").read_text())["output"]["theta_final"])
    tb = np.array(json.loads((b / "results.json").read_text())["output"]["theta_final"])
    return np.max(np.abs(ta - tb))






def fmt_time(seconds):
    """Readable duration, or 'not logged'."""
    if seconds is None:
        return "not logged"
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


# Sort the runs into old (qiskit) runs and hybrid runs.
old_runs, hybrid_runs = {}, []
for folder in sorted(RESULTS.iterdir()):
    if not (folder / "results.json").exists():
        continue
    settings, backend = read_settings(folder)
    if backend == "hybrid":
        hybrid_runs.append((settings, folder))
    else:
        old_runs[settings] = folder

times = read_elapsed_times()

print(f"{'events':>6} {'epochs':>6} {'b':>5} {'alpha':>6} {'scaling':>7} | "
      f"{'history':>8} {'p_val':>8} {'theta':>8} | {'old time':>10} {'new time':>10}")
all_ok = True
for settings, new in hybrid_runs:
    circuit, layers, n_events, n_iter, b, alpha, scaling = settings
    old = old_runs.get(settings)
    if old is None:
        print(f"{n_events:>6} {n_iter:>6} {b:>5} {alpha:>6} {str(scaling):>7} | no old run with these settings")
        continue
    d_hist = max_diff_history(old, new)
    d_p    = max_diff_predictions(old, new)
    d_th   = max_diff_theta(old, new)
    if d_hist > 2e-6 or d_p > 2e-6 or d_th > 1e-6:
        all_ok = False
    print(f"{n_events:>6} {n_iter:>6} {b:>5} {alpha:>6} {str(scaling):>7} | "
          f"{d_hist:>8.1e} {d_p:>8.1e} {d_th:>8.1e} | "
          f"{fmt_time(times.get(old.name)):>10} {fmt_time(times.get(new.name)):>10}")

print()
print("All hybrid runs identical to the old runs (up to rounding)." if all_ok
      else "WARNING: at least one hybrid run differs from its old run.")