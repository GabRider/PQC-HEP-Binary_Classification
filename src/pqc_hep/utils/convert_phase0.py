"""
Convert a run folder of the phase-0 script (PQC_phase0.py) into a run folder of
main.py: the missing files are written into the same folder, so old runs are
read and reused like new ones.

Usage, from src/pqc_hep (the data paths are relative, like in example.toml):
    python -m pqc_hep.utils.convert_phase0 <run_folder>              # config.toml
    python -m pqc_hep.utils.convert_phase0 <run_folder> --name scan  # scan.toml

Every setting comes from the run's results.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pqc_hep import circuits, output
from pqc_hep.config import Config
from pqc_hep.data import N_FEAT, Events
from pqc_hep.quantum import QModel


def eval_points(checkpoints: dict) -> list:
    """batch_at of the passes in a results.json checkpoint dict:
    {"start": ..., "10": ...} -> ["start", 10]."""
    return [key if key == "start" else int(key) for key in checkpoints]


def config_from_phase0(results: dict, n_val: int) -> Config:
    """Settings of the phase-0 run."""
    info = results["run_info"]
    data_folder = Path(info["data_folder"])
    batch_size = info["BATCH_SIZE"]
    if not info["BATCH_SIZE_IS_FRACTION"]:
        batch_size = int(batch_size)
    seeds = {
        "data_shuffle": info["DATA_SHUFFLE_SEED"],
        "theta_init": info["THETA_INIT_SEED"],
        "pass_reshuffle": info["PASS_RESHUFFLE_SEED"],
        "simulator": info["SIM_SEED"],
        # phase 0 draws the validation subset with the data shuffle seed
        "validation": info["DATA_SHUFFLE_SEED"],
    }
    return Config.model_validate(
        {
            "data": {
                "sig_path": data_folder / info["signal_file"],
                "bkg_path": data_folder / info["background_file"],
                # Not recorded by phase 0: its default file names.
                "val_sig_path": data_folder / "val_sig.csv",
                "val_bkg_path": data_folder / "val_bkg.csv",
                "max_events": info["max_events_limit"] or "all",
                "balance_classes": info["BALANCE_CLASSES"],
                "event_order": info["EVENT_ORDER"],
                "reshuffle_between_passes": info["RESHUFFLE_BETWEEN_PASSES"],
                "feature_scaling": info["FEATURE_SCALING"] or "none",
                "n_val": n_val,
            },
            "model": {
                "circuit": {
                    "circuit_number": results["circuit"]["circuit_number"],
                    "n_layers": results["circuit"]["n_layers"],
                },
                "training": {"n_iter": info["N_ITER"], "batch_size": batch_size},
                "adam": {
                    "alpha": info["ALPHA"],
                    "beta1": info["BETA1"],
                    "beta2": info["BETA2"],
                    "epsilon": info["EPSILON"],
                },
                "evaluation": {
                    # phase 0 measures the train and validation metrics at
                    # start and after every epoch (history.csv)
                    "global_at": ["start", "end"],
                    "global_step": 1,
                    "batch_at": eval_points(
                        results["output"]["batch_eval_checkpoints"]
                    ),
                },
                "backend": {
                    "use_statevector": info["USE_STATEVECTOR"],
                    "nm": info["Nm"],
                    "simulator_method": info["SIMULATOR_METHOD"],
                    "ibm_backend": info["IBM_BACKEND"],
                },
            },
            # A null seed was drawn at random and not recorded: left out.
            "seeds": {name: seed for name, seed in seeds.items() if seed is not None},
        }
    )


def convert(run_dir: Path, name: str | None) -> None:
    """Write model.json, <name>.toml (config.toml without a name) and
    classifier.py into the phase-0 run folder run_dir."""
    results = json.loads((run_dir / "results.json").read_text())
    info = results["run_info"]
    # results.json does not record n_val: one line per validation event.
    n_val = len((run_dir / "val_predictions.csv").read_text().splitlines()) - 1
    cfg = config_from_phase0(results, n_val)

    circuit = circuits.CIRCUITS[cfg.model.circuit.circuit_number]
    qmodel = QModel(circuit, cfg.model.circuit.n_layers, backend=None, shots=None)
    gates = qmodel.gate_sequence(results["output"]["theta_final"])
    zscore = None
    if info["FEATURE_SCALING"] == "zscore":
        zscore = {"mean": info["feature_mean"], "std": info["feature_std"]}
    # The old run recorded no examples: the classifier self-test has none to check.
    no_examples = Events(np.empty((0, N_FEAT)), np.empty(0), np.empty(0))
    model = output.build_model(
        cfg, info["timestamp"], gates, no_examples, np.empty(0), zscore
    )
    # results.json is rewritten unchanged; history.csv and val_predictions.csv
    # already have the format of main.py.
    config_path = f"{name}.toml" if name is not None else None
    output.save(run_dir, results, model, cfg, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a phase-0 run folder into a main.py run folder."
    )
    parser.add_argument("run_dir", type=Path, help="folder written by PQC_phase0.py")
    parser.add_argument(
        "--name", default=None, help="name of the TOML written (default: config)"
    )
    args = parser.parse_args()
    # Errors caused by the user's files: a message without traceback.
    try:
        convert(args.run_dir, args.name)
    except (OSError, ValueError) as e:
        sys.exit(f"Error: {e}")
