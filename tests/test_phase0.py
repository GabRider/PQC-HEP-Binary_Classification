"""
Equivalence with the phase-0 script: run reference/phase0.py (seeded copy) and
main.py with the same settings, then compare theta, the learning curves, the
validation predictions and the classifier. Also checks convert_phase0.py, which
turns a phase-0 run folder into a main.py run folder.

Excluded by default; run it with:
    uv run pytest -m phase0
"""

import csv
import tomllib

import pytest
import tomli_w
from helpers import (
    DATA,
    PKG,
    Reference,
    classifier_events,
    main_config,
    needs_data,
    predict,
    results_output,
    run,
    train,
    variants,
)

# Statevector only: phase0.py runs more circuits (validation) between the
# training steps, so in shot mode the simulator draws would not line up.
BASE = {
    "max_events": 40,
    "balance_classes": False,
    "event_order": "random",
    "reshuffle_between_passes": True,
    "n_iter": 2,
    "batch_size": 10,
    "use_statevector": True,
    "circuit_number": 1,
    "feature_scaling": "zscore",
    "n_val": 20,
}
CASES = {
    "zscore": BASE,
    "no_scaling": {**BASE, "feature_scaling": "none"},
    **variants(BASE),
}


def phase0_config(case):
    """main_config plus the phase-0 settings. phase0.py draws the validation
    subset with its data shuffle seed (11) and measures the train and validation
    metrics after every epoch (global_step = 1)."""
    config = main_config(case)
    config["data"] |= {
        "feature_scaling": case["feature_scaling"],
        "val_sig_path": str(DATA / "val_sig.csv"),
        "val_bkg_path": str(DATA / "val_bkg.csv"),
        "n_val": case["n_val"],
    }
    config["model"]["evaluation"] = {"global_step": 1}
    config["seeds"]["validation"] = 11
    return config


def read_csv(path):
    """Rows of a CSV file, every value as a float."""
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def assert_same_csv(new, old):
    # phase0.py rounds to 6 decimals (weights: 6 significant digits).
    assert len(new) == len(old)
    for new_row, old_row in zip(new, old, strict=True):
        assert new_row == pytest.approx(old_row, rel=1e-5, abs=1e-6)


@needs_data
@pytest.mark.phase0
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_same_output_as_phase0(tmp_path, case):
    old, new = train(tmp_path, case, phase0_config(case), Reference.PHASE0)
    assert results_output(new)["theta_final"] == results_output(old)["theta_final"]
    for name in ("history.csv", "val_predictions.csv"):
        assert_same_csv(read_csv(new / name), read_csv(old / name))
    events = classifier_events(tmp_path)
    assert predict(new, events) == predict(old, events)


@needs_data
@pytest.mark.phase0
def test_converted_phase0_run(tmp_path):
    # phase0.py run, converted in place, then rerun by main.py with the
    # converted TOML.
    old_dir = tmp_path / "old"
    rerun_dir = tmp_path / "rerun"
    old_dir.mkdir()
    run(Reference.PHASE0.args, old_dir, BASE)
    old = next(old_dir.glob("*/"))
    events = classifier_events(tmp_path)
    phase0_predictions = predict(old, events)  # before its classifier.py is replaced

    # From src/pqc_hep: the converted data paths are relative to it (Data/...).
    run(["-m", "pqc_hep.utils.convert_phase0", old], PKG, {})
    config = tomllib.loads((old / "config.toml").read_text())
    config["data"]["results_dir"] = str(rerun_dir / "result")
    rerun_dir.mkdir()
    (rerun_dir / "config.toml").write_text(tomli_w.dumps(config))
    run([PKG / "main.py", "--config", rerun_dir / "config.toml"], PKG, {})
    new = next((rerun_dir / "result").glob("*/*/*/"))

    # Same files as a main.py run.
    assert {f.name for f in old.iterdir()} == {f.name for f in new.iterdir()}
    # The converted TOML reruns the same training.
    assert results_output(new)["theta_final"] == results_output(old)["theta_final"]
    assert_same_csv(read_csv(new / "history.csv"), read_csv(old / "history.csv"))
    # The converted model.json gives the phase-0 classifier's predictions.
    assert predict(old, events) == phase0_predictions
