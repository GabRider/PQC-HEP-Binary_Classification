"""
Equivalence with Adriano's original script: run original.py (seeded copy) and
main.py with the same settings, then compare their outputs.

Both tests are excluded by default; run them with:
    uv run pytest -m refactoring   # theta_final and every loss, one run per CASES
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
CIRCUITS = TESTS / "original_circuits.py"

# Settings given to both scripts. Keys are the TOML names of main.py;
# original.py reads each one from the PQC_<KEY> environment variable.
BASE = {
    "max_events": 40,
    "balance_classes": False,
    "event_order": "random",
    "reshuffle_between_passes": True,
    "n_iter": 2,
    "batch_size": 10,
    "use_statevector": False,
    "circuit_number": 1,  # RX/RZ only: two-term shift rule
}
CASES = {
    "default": BASE,
    "balance": {**BASE, "balance_classes": True},
    # reshuffle_between_passes needs event_order = "random"
    "signal_first": {
        **BASE,
        "event_order": "signal_first",
        "reshuffle_between_passes": False,
    },
    "background_first": {
        **BASE,
        "event_order": "background_first",
        "reshuffle_between_passes": False,
    },
    # CRZ followed by RX-RZ: the four-term rule's ±π shifts change the output
    "four_term": {**BASE, "circuit_number": 7},
}

# main.py config; {...} come from a case. Fixed values match original.py.
CONFIG = """
[data]
sig_path = "{data}/train_sig.csv"
bkg_path = "{data}/train_bkg.csv"
max_events = {max_events}
balance_classes = {balance_classes}
event_order = {event_order}
reshuffle_between_passes = {reshuffle_between_passes}

[model.circuit]
circuit_number = {circuit_number}

[model.training]
n_iter = {n_iter}
batch_size = {batch_size}

[model.backend]
nm = 100
use_statevector = {use_statevector}

[seeds]
data_shuffle = 11
theta_init = 22
pass_reshuffle = 33
simulator = 44
"""
ORIGINAL_ARGS = ["--circuits-file", CIRCUITS, "--data-folder", DATA]

needs_data = pytest.mark.skipif(not DATA.is_dir(), reason=f"no data in {DATA}")


def toml(value):
    """Convert bools to strings and quote strings for TOML; leave numbers as-is."""
    if isinstance(value, bool):
        return str(value).lower()
    return f'"{value}"' if isinstance(value, str) else str(value)


def run(args, cwd, case):
    """Run the original.py with custom arguments"""
    env = {**os.environ, **{f"PQC_{k.upper()}": str(v) for k, v in case.items()}}
    subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, check=True, capture_output=True
    )


def train(tmp_path, case):
    """Train with original.py and main.py; return their two output folders."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    config = CONFIG.format(data=DATA, **{k: toml(v) for k, v in case.items()})
    (new / "config.toml").write_text(config)
    run([TESTS / "original.py", *ORIGINAL_ARGS], old, case)
    run([PKG / "main.py", "--config", "config.toml"], new, case)
    return next(old.glob("*/")), next((new / "result").glob("*/*/*/"))


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
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_same_output_as_original(tmp_path, case):
    old, new = train(tmp_path, case)
    assert results_output(new) == results_output(old)


@needs_data
@pytest.mark.classifier
def test_classifier_same_predictions_as_original(tmp_path):
    # Statevector mode: the original classifier is deterministic only there.
    old, new = train(tmp_path, {**BASE, "use_statevector": True})
    sig = (DATA / "train_sig.csv").read_text().splitlines()[:10]
    bkg = (DATA / "train_bkg.csv").read_text().splitlines()[:10]
    events = tmp_path / "events.csv"
    events.write_text("\n".join(sig + bkg) + "\n")
    assert predict(new, events) == predict(old, events)
