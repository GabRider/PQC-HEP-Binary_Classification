"""
Equivalence with Adriano's original script: run reference/original.py (seeded
copy) and main.py with the same settings, then compare their outputs.

Both tests are excluded by default; run them with:
    uv run pytest -m refactoring   # theta_final and every loss, one run per CASES
    uv run pytest -m classifier    # classifier.py predictions (statevector mode)
"""

import pytest
from helpers import (
    Reference,
    classifier_events,
    main_config,
    needs_data,
    predict,
    results_output,
    train,
    variants,
)

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
CASES = {"default": BASE, **variants(BASE)}


@needs_data
@pytest.mark.refactoring
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_same_output_as_original(tmp_path, case):
    old, new = train(tmp_path, case, main_config(case), Reference.ORIGINAL)
    assert results_output(new) == results_output(old)


@needs_data
@pytest.mark.classifier
def test_classifier_same_predictions_as_original(tmp_path):
    # Statevector mode: the original classifier is deterministic only there.
    case = {**BASE, "use_statevector": True}
    old, new = train(tmp_path, case, main_config(case), Reference.ORIGINAL)
    events = classifier_events(tmp_path)
    assert predict(new, events) == predict(old, events)
