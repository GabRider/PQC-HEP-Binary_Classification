"""
feature_amplitude_shares.py - share of each feature in the amplitude-encoded state.

With amplitude encoding, each event x is divided by its norm: c = x / |x|. This script
prints the mean of c_k^2 (in %) for each of the 12 features, for raw features, min-max
scaling to [-1, 1] and z-score standardisation. Without scaling, the four GeV-scale
variables carry ~99 % of the total.

Usage (from anywhere):  python phase0/analysis/feature_amplitude_shares.py
"""

from pathlib import Path
import numpy as np

DATA = Path(__file__).parent.parent.parent / "data"      # data/ at the repository root

NAMES = ["Q(l)", "pT(l)", "eta(l)", "MET", "MT", "N(jet)",
         "pT(jet1)", "HT", "N(b)", "pT(jet b)", "CSV", "dR(l,b)"]

def read(file):
    """Columns 0-11: the 12 features (weight and process name are ignored)."""
    return np.genfromtxt(file, delimiter=",", usecols=range(12))

sig = read(DATA / "train_sig.csv")
bkg = read(DATA / "train_bkg.csv")
X = np.vstack([sig, bkg])
print(f"{len(sig)} signal + {len(bkg)} background events = {len(X)}\n")

def shares(A):
    """Mean squared amplitude of each feature, in % (the 4 zero-padding entries do not change the norm)."""
    c = A / np.linalg.norm(A, axis=1, keepdims=True)
    return 100 * (c ** 2).mean(axis=0)

lo, hi = X.min(axis=0), X.max(axis=0)
X_minmax = (2 * X - hi - lo) / (hi - lo)               # min-max scaling to [-1, 1]
X_z = (X - X.mean(axis=0)) / X.std(axis=0)             # z-score standardisation

s_raw, s_mm, s_z = shares(X), shares(X_minmax), shares(X_z)

print(f"{'Feature':10s} {'raw':>8s} {'[-1,1]':>8s} {'z-score':>8s}")
for n, a, b, c in zip(NAMES, s_raw, s_mm, s_z):
    print(f"{n:10s} {a:7.2f}% {b:7.2f}% {c:7.2f}%")

top4 = sorted(s_raw)[-4:]
print(f"\nRaw: 4 largest = {sum(top4):.1f} %, other 8 = {100 - sum(top4):.1f} %")
