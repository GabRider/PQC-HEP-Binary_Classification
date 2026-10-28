"""Run outputs: results.json, model.json, effective TOML, classifier.py copy,
history.csv, val_predictions.csv and the final summary."""

import csv
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tomli_w

from pqc_hep.circuits import CIRCUITS
from pqc_hep.config import Config
from pqc_hep.data import N_FEAT, Events
from pqc_hep.quantum import N_QUBITS
from pqc_hep.training import EvalSchedule, Evaluation


def _short_count(n: int) -> str:
    """Event count for a folder name: 10000 -> "10k", 1500 -> "1500"."""
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def run_dir_name(
    cfg: Config,
    config_path: str | None,
    n_events: int,
    batch_size: int,
    now: datetime.datetime,
) -> Path:
    """Run folder results_dir/c<circuit>/<date>/<time>_<toml>_<settings>, e.g.
    result/c01/2026-10-23/11h20m01_example_L1_e10_train1k_b100_lr0.005_seed3f9a2c.
    The time comes first so a day's runs sort in order; <toml> is the
    --config file name, absent without one."""
    c = cfg.model.circuit
    # Same 4 seeds -> same id. hashlib, not hash(): stable across Python versions.
    seeds = json.dumps(cfg.seeds.model_dump(), sort_keys=True).encode()
    seed_id = hashlib.sha1(seeds).hexdigest()[:6]
    has_val = cfg.data.val_sig_path is not None
    config_name = [Path(config_path).stem] if config_path is not None else []
    val_part = [f"val{_short_count(cfg.data.n_val)}"] if has_val else []
    # config_name and val_part are optional: an empty list unpacked with * adds
    # nothing. :g drops trailing zeros and uses scientific notation if needed.
    parts = [
        now.strftime("%Hh%Mm%S"),
        *config_name,
        f"L{c.n_layers}",
        f"e{cfg.model.training.n_iter}",
        f"train{_short_count(n_events)}",
        *val_part,
        f"b{batch_size}",
        f"lr{cfg.model.adam.alpha:g}",
        f"seed{seed_id}",
    ]
    return (
        cfg.data.results_dir
        / f"c{c.circuit_number:02d}"
        / now.strftime("%Y-%m-%d")
        / "_".join(parts)
    )


def _theta_labels(gates: list[dict]) -> list[str]:
    """A human-readable label for each parameterised gate in the variational form."""
    out = []
    for gate in gates:
        if gate["params"]:
            name = gate["name"]
            qubits = gate["qubits"]
            val = gate["params"][0]
            if name in ("crz", "crx"):
                out.append(
                    f"CTRL-{name[1:].upper()}(ctrl=q{qubits[0]}, tgt=q{qubits[1]})"
                    f" = {val:.6f} rad"
                )
            else:
                out.append(f"{name.upper()}(q{qubits[0]}) = {val:.6f} rad")
    return out


# TODO: refactor when main.py is clean (12 arguments).
def build_results(
    cfg: Config,
    config_path: str | None,
    timestamp: str,
    n_signal: int,
    n_background: int,
    n_events: int,
    batch_size: int,
    eval_schedule: EvalSchedule,
    theta_final: list[float],
    gates: list[dict],
    evals: list[Evaluation],
) -> dict:
    """Content of results.json: run settings, circuit, trained theta and evaluations."""
    c = cfg.model.circuit
    return {
        "run_info": {
            "signal_file": cfg.data.sig_path.name,
            "background_file": cfg.data.bkg_path.name,
            "data_folder": cfg.data.sig_path.resolve().parent.name,
            "timestamp": timestamp,
            "n_signal": n_signal,
            "n_background": n_background,
            "n_events_loaded": n_signal + n_background,
            "n_events_used": n_events,
            "max_events_limit": cfg.data.event_limit,
            "BALANCE_CLASSES": cfg.data.balance_classes,
            "EVENT_ORDER": cfg.data.event_order,
            "DATA_SHUFFLE_SEED": cfg.seeds.data_shuffle,
            "RESHUFFLE_BETWEEN_PASSES": cfg.data.reshuffle_between_passes,
            "PASS_RESHUFFLE_SEED": cfg.seeds.pass_reshuffle,
            "N_ITER": cfg.model.training.n_iter,
            "BATCH_SIZE_IS_FRACTION": isinstance(cfg.model.training.batch_size, float),
            "BATCH_SIZE": batch_size,
            "N_BATCHES_PER_ITER": n_events // batch_size,
            "ALPHA": cfg.model.adam.alpha,
            "BETA1": cfg.model.adam.beta1,
            "BETA2": cfg.model.adam.beta2,
            "EPSILON": cfg.model.adam.epsilon,
            "adam_formulation": "sklearn_AdamOptimizer_Kingma_Ba_2014",
            "THETA_INIT_SEED": cfg.seeds.theta_init,
            "GLOBAL_EVAL_CHECKPOINTS": sorted(eval_schedule.global_passes),
            "GLOBAL_EVAL_STEP": cfg.model.evaluation.global_step,
            "BATCH_EVAL_CHECKPOINTS": sorted(eval_schedule.batch_passes),
            "BATCH_EVAL_STEP": cfg.model.evaluation.batch_step,
            "USE_STATEVECTOR": cfg.model.backend.use_statevector,
            "SIMULATOR_METHOD": cfg.model.backend.simulator_method,
            "Nm": cfg.model.backend.nm,
            "SIM_SEED": cfg.seeds.simulator,
            "IBM_BACKEND": cfg.model.backend.ibm_backend,
            "CONFIG_FILE": config_path,
        },
        "circuit": {
            "circuit_number": c.circuit_number,
            "circuit_description": CIRCUITS[c.circuit_number].description,
            "n_layers": c.n_layers,
            "total_params": len(theta_final),
        },
        "output": {
            "theta_final": theta_final,
            "theta_labels": _theta_labels(gates),
            "global_eval_checkpoints": {
                e.key: e.train.loss for e in evals if e.train is not None
            },
            "batch_eval_checkpoints": {
                e.key: e.batch_loss for e in evals if e.batch_loss is not None
            },
        },
    }


def build_model(
    cfg: Config,
    timestamp: str,
    gates: list[dict],
    examples: Events,
    example_p: np.ndarray,
    zscore: dict | None,
) -> dict:
    """Content of model.json. classifier.py rebuilds the circuit from it; the
    examples are training events with their final P(signal), so running
    classifier.py alone checks that it reproduces them."""
    return {
        "timestamp": timestamp,
        "circuit": {
            **cfg.model.circuit.model_dump(mode="json"),
            "description": CIRCUITS[cfg.model.circuit.circuit_number].description,
        },
        "n_qubits": N_QUBITS,
        "n_feat": N_FEAT,
        "zscore": zscore,  # {"mean": [...], "std": [...]}, or null without z-score
        "gates": gates,
        "backend": {
            **cfg.model.backend.model_dump(mode="json"),
            "sim_seed": cfg.seeds.simulator,
        },
        "examples": [
            {"features": x.tolist(), "label": int(y), "expected_p": float(p)}
            for x, y, p in zip(
                examples.features, examples.labels, example_p, strict=True
            )
        ],
    }


def save(
    output_dir: str, results: dict, model: dict, cfg: Config, config_path: str | None
) -> None:
    """Write results.json, model.json, the effective configuration (named like
    the --config file, config.toml without one) and a copy of classifier.py to
    output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}/")

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON  : {json_path}")

    model_path = os.path.join(output_dir, "model.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"  Model JSON    : {model_path}")

    # Effective configuration: every value, seeds drawn at random included, so
    # `--config` on this file reruns the same training. TOML has no null, so the
    # None values (= default) are left out.
    config_name = Path(config_path).name if config_path else "config.toml"
    toml_path = os.path.join(output_dir, config_name)
    with open(toml_path, "wb") as f:
        tomli_w.dump(cfg.model_dump(mode="json", exclude_none=True), f)
    print(f"  Config TOML   : {toml_path}")

    classifier_path = shutil.copy(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier.py"),
        output_dir,
    )
    print(f"  Classifier    : {classifier_path}")


HISTORY_COLUMNS = ["epoch", "step", "train_loss", "train_auc", "val_loss", "val_auc"]


def save_history(
    output_dir: str, evals: list[Evaluation], n_batches_per_iter: int
) -> None:
    """history.csv: one line per global evaluation (epoch 0 = start)."""
    path = os.path.join(output_dir, "history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for e in evals:
            if e.train is not None:  # skip the evaluations with only a batch loss
                writer.writerow(e.as_row(n_batches_per_iter))
    print(f"  History CSV   : {path}")


def save_val_predictions(output_dir: str, val: Events, p: np.ndarray) -> None:
    """val_predictions.csv: label, weight and final p of each validation event."""
    path = os.path.join(output_dir, "val_predictions.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "weight", "p"])
        writer.writerows(zip(val.labels.astype(int), val.weights, p, strict=True))
    print(f"  Val preds CSV : {path}")


def print_summary(results: dict, output_dir: str, elapsed: float) -> None:
    """Final console summary, read from the results.json content."""
    info, circ, out = results["run_info"], results["circuit"], results["output"]
    n_events = info["n_events_used"]

    print("\nTraining complete.")
    print(f"  Circuit      : [{circ['circuit_number']}] {circ['circuit_description']}")
    print(f"  Layers       : {circ['n_layers']}  |  Parameters: {circ['total_params']}")
    print(f"  Events       : {n_events} used of {info['n_events_loaded']} loaded")
    print(
        f"  N_iter       : {info['N_ITER']}"
        f"  |  Batches/iter: {info['N_BATCHES_PER_ITER']}"
        f"  |  Batch size: {info['BATCH_SIZE']}"
    )
    print(f"  theta_final  : {np.round(out['theta_final'], 4)}")

    for name, key in (("global", "global"), ("batch ", "batch")):
        passes = info[f"{key.upper()}_EVAL_CHECKPOINTS"]
        if passes:
            n = passes[-1]
            loss = out[f"{key}_eval_checkpoints"][str(n)]
            print(f"  Last {name} eval (n={n}): loss={loss:.6f}")
    print(f"  Output folder: {output_dir}/")

    sim_info = (
        "statevector (exact)" if info["USE_STATEVECTOR"] else f"Nm={info['Nm']} shots"
    )
    print(f"\n  Elapsed time : {elapsed:.1f} s  ({elapsed / 60:.1f} min)")
    print(
        f"  Per event    : {elapsed / n_events:.3f} s/event  "
        f"(N_iter={info['N_ITER']}, {sim_info}, {n_events} events)"
    )
