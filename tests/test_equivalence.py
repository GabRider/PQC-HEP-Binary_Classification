"""
Equivalence with Adriano's original script: run original.py (seeded copy) and
main.py with the same settings, then compare their outputs.

Both tests are excluded by default; run them with:
    uv run pytest -m refactoring   # theta_final and every loss (shot-based mode)
    uv run pytest -m classifier    # classifier.py predictions (statevector mode)
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
PKG = TESTS.parent / "src" / "pqc_hep"
DATA = PKG / "Data"
CIRCUITS = PKG / "circuits_library_Qiskit.py"

# Same values as the fixed settings in original.py.
CONFIG = """
[data]
sig_path = "{data}/train_sig.csv"
bkg_path = "{data}/train_bkg.csv"
max_events = 200
reshuffle_between_passes = true

[model.circuit]
circuits_file = "{circuits}"

[model.training]
n_iter = 3
batch_size = 50

[model.backend]
nm = 100
use_statevector = {statevector}

[seeds]
data_shuffle = 11
theta_init = 22
pass_reshuffle = 33
simulator = 44
"""
ORIGINAL_ARGS = [
    *("--circuits-file", CIRCUITS, "--data-folder", DATA),
    *("--max-events", "200", "--n-iter", "3", "--batch-size", "50"),
]

needs_data = pytest.mark.skipif(not DATA.is_dir(), reason=f"no data in {DATA}")


def run(args, cwd, statevector):
    """Run a script; original.py reads the mode from PQC_STATEVECTOR."""
    env = {**os.environ, "PQC_STATEVECTOR": "1" if statevector else "0"}
    subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, check=True, capture_output=True
    )


def train(tmp_path, statevector):
    """Train with original.py and main.py; return their two output folders."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    config = CONFIG.format(
        data=DATA, circuits=CIRCUITS, statevector=str(statevector).lower()
    )
    (new / "config.toml").write_text(config)
    run([TESTS / "original.py", *ORIGINAL_ARGS], old, statevector)
    run([PKG / "main.py", "--config", "config.toml"], new, statevector)
    return next(old.glob("*/")), next(new.glob("*/"))


def results_output(run_dir):
    return json.loads((run_dir / "results.json").read_text())["output"]


def predict(run_dir, events):
    return subprocess.run(
        [sys.executable, "classifier.py", events],
        cwd=run_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@needs_data
@pytest.mark.refactoring
def test_same_output_as_original(tmp_path):
    old, new = train(tmp_path, statevector=False)
    assert results_output(new) == results_output(old)


@needs_data
@pytest.mark.classifier
def test_classifier_same_predictions_as_original(tmp_path):
    # Statevector mode: the original classifier is deterministic only there.
    old, new = train(tmp_path, statevector=True)
    sig = (DATA / "train_sig.csv").read_text().splitlines()[:10]
    bkg = (DATA / "train_bkg.csv").read_text().splitlines()[:10]
    events = tmp_path / "events.csv"
    events.write_text("\n".join(sig + bkg) + "\n")
    assert predict(new, events) == predict(old, events)
