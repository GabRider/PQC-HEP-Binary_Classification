"""Run outputs: results.json, model.json, classifier.py copy, final summary."""

import json
import os
import shutil

import numpy as np

from pqc_hep.circuits import CIRCUITS
from pqc_hep.config import Config
from pqc_hep.data import N_FEAT, Events
from pqc_hep.quantum import N_QUBITS
from pqc_hep.training import EvalSchedule


def run_dir_name(cfg: Config, batch_size: int, timestamp: str) -> str:
    """Folder named after the circuit, layers, training settings, and run time."""
    c = cfg.model.circuit
    name = (
        f"circuit{c.circuit_number}_layers{c.n_layers}_"
        f"{cfg.model.training.n_iter}iter_batch{batch_size}_{timestamp}"
    )
    return f"{cfg.data.run_tag}_{name}" if cfg.data.run_tag else name


def _theta_labels(gates: list[dict]) -> list[str]:
    """A human-readable label for each parameterised gate in the variational form."""
    out = []
    for gate in gates:
        if gate["params"]:
            name = gate["name"]
            qubits = gate["qubits"]
            val = gate["params"][0]
            if name in ("crz", "crx", "cry"):
                out.append(
                    f"CTRL-{name[1:].upper()}(ctrl=q{qubits[0]}, tgt=q{qubits[1]})"
                    f" = {val:.6f} rad"
                )
            else:
                out.append(f"{name.upper()}(q{qubits[0]}) = {val:.6f} rad")
    return out


# TODO: refactor when main.py is clean (11 arguments).
def build_results(
    cfg: Config,
    timestamp: str,
    n_signal: int,
    n_background: int,
    n_events: int,
    batch_size: int,
    eval_schedule: EvalSchedule,
    theta_final: list[float],
    gates: list[dict],
    global_evals: dict[str, float],
    batch_evals: dict[str, float],
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
            "max_events_limit": (
                None if cfg.data.max_events == "all" else cfg.data.max_events
            ),
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
            "RUN_TAG": cfg.data.run_tag,
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
            "global_eval_checkpoints": global_evals,
            "batch_eval_checkpoints": batch_evals,
        },
    }


def build_model(
    cfg: Config,
    timestamp: str,
    gates: list[dict],
    examples: Events,
    example_p: np.ndarray,
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


def save(output_dir: str, results: dict, model: dict) -> None:
    """Write results.json, model.json and a copy of classifier.py to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving outputs to: {output_dir}/")

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [1/3] Results JSON : {json_path}")

    model_path = os.path.join(output_dir, "model.json")
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)
    print(f"  [2/3] Model JSON   : {model_path}")

    classifier_path = shutil.copy(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier.py"),
        output_dir,
    )
    print(f"  [3/3] Classifier   : {classifier_path}")


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
