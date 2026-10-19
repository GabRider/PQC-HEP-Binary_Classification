"""
Equivalence with Adriano's original script: run original.py (seeded copy) and
main.py with the same settings, then compare theta_final and every loss.

Excluded by default; run it with: uv run pytest -m refactoring
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.refactoring

TESTS = Path(__file__).parent
PKG = TESTS.parent / "src" / "pqc_hep"
DATA = PKG / "Data"
CIRCUITS = PKG / "circuits_library_Qiskit.py"

# Same values as the fixed settings in original.py.
CONFIG = f"""
[data]
sig_path = "{DATA / "train_sig.csv"}"
bkg_path = "{DATA / "train_bkg.csv"}"
max_events = 200
reshuffle_between_passes = true

[model.circuit]
circuits_file = "{CIRCUITS}"

[model.training]
n_iter = 3
batch_size = 50

[model.backend]
nm = 100

[seeds]
data_shuffle = 11
theta_init = 22
pass_reshuffle = 33
simulator = 44
"""


def run(args, cwd):
    subprocess.run([sys.executable, *args], cwd=cwd, check=True, capture_output=True)
    return json.loads(next(cwd.glob("*/results.json")).read_text())["output"]


def test_same_output_as_original(tmp_path):
    if not DATA.is_dir():
        pytest.skip(f"data files not found in {DATA}")
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (new / "config.toml").write_text(CONFIG)

    expected = run(
        [
            TESTS / "original.py",
            "--circuits-file",
            CIRCUITS,
            "--data-folder",
            DATA,
            "--max-events",
            "200",
            "--n-iter",
            "3",
            "--batch-size",
            "50",
        ],
        old,
    )
    assert run([PKG / "main.py", "--config", "config.toml"], new) == expected
