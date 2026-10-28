"""Shared by the tests that compare main.py with a reference script of
tests/reference/: the common cases and config, running both scripts with the
same settings, then reading their outputs."""

import json
import os
import subprocess
import sys
from enum import Enum
from pathlib import Path

import pytest
import tomli_w

TESTS = Path(__file__).parent
REFERENCE = TESTS / "reference"
PKG = TESTS.parent / "src" / "pqc_hep"
DATA = PKG / "Data"

needs_data = pytest.mark.skipif(not DATA.is_dir(), reason=f"no data in {DATA}")


def variants(base):
    """The cases shared by every equivalence test, built on `base`."""
    return {
        "balance": {**base, "balance_classes": True},
        # reshuffle_between_passes needs event_order = "random"
        "signal_first": {
            **base,
            "event_order": "signal_first",
            "reshuffle_between_passes": False,
        },
        "background_first": {
            **base,
            "event_order": "background_first",
            "reshuffle_between_passes": False,
        },
        # CRZ followed by RX-RZ: the four-term rule's ±π shifts change the output
        "four_term": {**base, "circuit_number": 7},
    }


def main_config(case):
    """main.py config of a case. The fixed values (shots, seeds) match the
    reference scripts."""
    return {
        "data": {
            "sig_path": str(DATA / "train_sig.csv"),
            "bkg_path": str(DATA / "train_bkg.csv"),
            "max_events": case["max_events"],
            "balance_classes": case["balance_classes"],
            "event_order": case["event_order"],
            "reshuffle_between_passes": case["reshuffle_between_passes"],
        },
        "model": {
            "circuit": {"circuit_number": case["circuit_number"]},
            "training": {"n_iter": case["n_iter"], "batch_size": case["batch_size"]},
            "backend": {"nm": 100, "use_statevector": case["use_statevector"]},
        },
        "seeds": {
            "data_shuffle": 11,
            "theta_init": 22,
            "pass_reshuffle": 33,
            "simulator": 44,
        },
    }


class Reference(Enum):
    """Reference scripts of tests/reference/ that main.py is compared with."""

    ORIGINAL = "original.py"
    PHASE0 = "phase0.py"

    @property
    def args(self) -> list:
        """Command-line arguments to run the script on the test data."""
        script = REFERENCE / self.value
        circuits = REFERENCE / "original_circuits.py"
        return [script, "--circuits-file", circuits, "--data-folder", DATA]


def run(args, cwd, case):
    """Run a script; the reference scripts read the case from PQC_<KEY> variables."""
    env = {**os.environ, **{f"PQC_{k.upper()}": str(v) for k, v in case.items()}}
    subprocess.run(
        [sys.executable, *args], cwd=cwd, env=env, check=True, capture_output=True
    )


def train(tmp_path, case, config, reference: Reference):
    """Train with the reference script and main.py; return their two output
    folders. config: main.py config (dict)."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (new / "config.toml").write_text(tomli_w.dumps(config))
    run(reference.args, old, case)
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


def classifier_events(tmp_path):
    """CSV of 10 signal and 10 background training events, raw features."""
    sig = (DATA / "train_sig.csv").read_text().splitlines()[:10]
    bkg = (DATA / "train_bkg.csv").read_text().splitlines()[:10]
    events = tmp_path / "events.csv"
    events.write_text("\n".join(sig + bkg) + "\n")
    return events
