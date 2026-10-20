"""
Author: Adriano Pinto Claro da Fonseca
Email: apcf@topfonseca.com
Institutional email: adriano.fonseca@tecnico.ulisboa.pt
Affiliation: Student at Instituto Superior Técnico, Universidade de Lisboa

 see Readme.md and adriano_code/README.md  for more information.
 Sphinx documentation will be implemented after refactoring.
"""

import sys
import time
import argparse
import os
import json
import datetime
import shutil
import numpy as np
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer

from tqdm import tqdm
from qiskit import QuantumCircuit

from pqc_hep import circuits
from pqc_hep.circuits import ShiftRule
from pqc_hep.config import load as load_config
from pqc_hep.data import (
    N_FEAT,
    load_events,
    normalise_weights,
    order_events,
    select_events,
    to_amplitudes,
)
from pqc_hep.quantum import N_QUBITS, QModel, make_backend
from pqc_hep.training import (
    EvalSchedule,
    Trainer,
    TrainingSet,
    WeightedBCE,
    measure_loss,
    resolve_batch_size,
)

_t_start = time.time()


# =============================================================================
# INPUT HANDLING
# =============================================================================
# Loads the TOML run configuration, resolves data file paths, loads the
# circuits library, and prepares the evaluation schedule.

parser = argparse.ArgumentParser(
    description="PQC mini-batch training pipeline for HEP binary classification (Qiskit)."
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="TOML configuration file; built-in defaults are used when omitted.",
)
args = parser.parse_args()

try:
    cfg = load_config(args.config)
except (OSError, TypeError, ValueError) as e:
    print(f"Error: {e}")
    sys.exit(1)

circuit_number = cfg.model.circuit.circuit_number
n_layers = cfg.model.circuit.n_layers
n_iter = cfg.model.training.n_iter
max_events = None if cfg.data.max_events == "all" else cfg.data.max_events
batch_size_is_fraction = isinstance(cfg.model.training.batch_size, float)
batch_size = (
    cfg.model.training.batch_size
)  # resolved to an event count after data loading

circuit = circuits.CIRCUITS[circuit_number]
shift_rules = circuit.shift_rules(n_layers)  # one per theta value
N_PARAMS = len(shift_rules)


eval_schedule = EvalSchedule.from_config(cfg.model.evaluation, n_iter)


# Print effective configuration.

_rule = "=" * 62
print(f"""
{_rule}
  EFFECTIVE CONFIGURATION
{_rule}
{cfg}
[extra]
  circuit                   : [{circuit_number}] {circuit.description}
  parameters                : {N_PARAMS}
  global_eval_at            : start={eval_schedule.global_start}, n_iter={sorted(eval_schedule.global_passes)}
  batch_eval_at             : start={eval_schedule.batch_start}, n_iter={sorted(eval_schedule.batch_passes)}
{_rule}
""")


# =============================================================================
# DATA LOADING
# =============================================================================
# Reads signal and background CSV files, selects and orders events, normalises
# weights relative to the selected training set, and validates batch_size.


print("Loading data...")
try:
    sig, bkg = load_events(cfg.data.sig_path, cfg.data.bkg_path)
    rng = np.random.default_rng(seed=cfg.seeds.data_shuffle)
    events = select_events(sig, bkg, max_events, cfg.data.balance_classes, rng)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)
events = order_events(events, cfg.data.event_order)
events = normalise_weights(events)
N_EVENTS = len(events)
N_LOADED = len(sig) + len(bkg)
n_sig, n_bkg = int((events.labels == 1).sum()), int((events.labels == 0).sum())

print(f"  Loaded : {N_LOADED} events ({len(sig)} signal, {len(bkg)} background)")
print(f"  Using  : {N_EVENTS} events ({n_sig} signal, {n_bkg} background)")
print(f"  Weight normalisation: sum(w_S) = sum(w_B) = {n_bkg}")

try:
    batch_size = resolve_batch_size(batch_size, N_EVENTS)
except ValueError as e:
    print(f"Error: {e}")
    sys.exit(1)
if batch_size_is_fraction:
    print(
        f"  Batch size (resolved) : {batch_size}  (= {batch_size/N_EVENTS*100:.1f}% of {N_EVENTS} events)"
    )

N_BATCHES_PER_ITER = N_EVENTS // batch_size
print(f"  Batches per N_iter : {N_BATCHES_PER_ITER}  (batch_size={batch_size})\n")


# =============================================================================
# QUANTUM PIPELINE
# =============================================================================
# The circuits and their execution are in quantum.py (QModel); the loss and the
# gradient via the parameter shift rule are in training.py.

# --- Backend -----------------------------------------------------------------
# Local AerSimulator, or a real IBM Quantum computer when ibm_backend is set.

try:
    backend, sampler = make_backend(cfg.model.backend, cfg.seeds.simulator)
except ImportError as e:
    print(f"Error: {e}")
    sys.exit(1)
if cfg.model.backend.ibm_backend is not None:
    print(f"  Using real IBM Quantum hardware: {cfg.model.backend.ibm_backend}")
shots = None if cfg.model.backend.use_statevector else cfg.model.backend.nm
qmodel = QModel(circuit, n_layers, backend, shots, sampler, preprocess=to_amplitudes)


# --- Output utilities --------------------------------------------------------


def record_final_circuit(theta_final):
    """Run the variational form once with the trained theta to record the gate sequence."""
    qc = QuantumCircuit(N_QUBITS)
    circuit.apply(qc, theta_final, n_layers)
    gates = []
    for instr in qc.data:
        gates.append(
            {
                "name": instr.operation.name,
                "qubits": [qc.find_bit(q).index for q in instr.qubits],
                "params": [float(p) for p in instr.operation.params],
            }
        )
    return gates


def generate_theta_labels(gate_sequence):
    """Return a human-readable label for each parameterised gate in the variational form."""
    out = []
    for gate in gate_sequence:
        if gate["params"]:
            name = gate["name"]
            qubits = gate["qubits"]
            val = gate["params"][0]
            if name in ("crz", "crx", "cry"):
                out.append(
                    f"CTRL-{name[1:].upper()}(ctrl=q{qubits[0]}, tgt=q{qubits[1]}) = {val:.6f} rad"
                )
            else:
                out.append(f"{name.upper()}(q{qubits[0]}) = {val:.6f} rad")
    return out


# =============================================================================
# TRAINING SETUP
# =============================================================================
# Build and transpile one full circuit per event (embedding + variational form).
# All N_iter passes reuse these transpiled circuits by binding parameter values.
print(f"Building and transpiling circuits ({N_EVENTS} events)...")
all_circuits = qmodel.build(events.features)
print(f"  Done: {len(all_circuits)} circuits transpiled.\n")

_n_two = shift_rules.count(ShiftRule.TWO_TERM)
_n_four = shift_rules.count(ShiftRule.FOUR_TERM)
print(
    f"  Shift rule classification: {_n_two} two-term (Rx/Ry/Rz), "
    f"{_n_four} four-term (CRx/CRy/CRz)\n"
)

# Evaluation losses, filled by record_eval().
global_eval_results = {}  # key: "start" or str(n_iter) -> loss value (float)
batch_eval_results = {}  # key: "start" or str(n_iter) -> loss value (float)


n_width = len(str(n_iter))
label_width = max(
    5, n_width
)  # "start" is 5 chars; ensures [n=start] aligns with [n=  N]


def record_eval(key, batch_loss, global_loss):
    """Save the losses of one evaluation ("start" or the pass number) and print
    them on one line; None = not measured at this evaluation."""
    if batch_loss is not None:
        batch_eval_results[key] = float(batch_loss)
    if global_loss is not None:
        global_eval_results[key] = float(global_loss)

    def show(loss):
        return "--------" if loss is None else f"{loss:.6f}"

    tqdm.write(
        f"  [n={key:>{label_width}}]  batch = {show(batch_loss)}   GLOBAL = {show(global_loss)}"
    )


print(
    f"Starting training: {n_iter} passes x {N_BATCHES_PER_ITER} batches "
    f"x {batch_size} events/batch"
)
print(
    f"  Simulation : {'statevector (exact)' if cfg.model.backend.use_statevector else f'Nm={cfg.model.backend.nm} shots'}"
)
print(
    f"  Adam       : sklearn AdamOptimizer, Kingma & Ba 2014  "
    f"alpha={cfg.model.adam.alpha}  beta1={cfg.model.adam.beta1}  beta2={cfg.model.adam.beta2}\n"
)


# =============================================================================
# TRAINING LOOP
# =============================================================================
# The config chooses the components; the Trainer only uses them.

# Initial theta uniformly in [-pi, pi].
theta0 = np.random.default_rng(seed=cfg.seeds.theta_init).uniform(-np.pi, np.pi, N_PARAMS)
optimizer = AdamOptimizer(
    params=[theta0],
    learning_rate_init=cfg.model.adam.alpha,
    beta_1=cfg.model.adam.beta1,
    beta_2=cfg.model.adam.beta2,
    epsilon=cfg.model.adam.epsilon,
)
loss = WeightedBCE()
trainer = Trainer(qmodel, loss, optimizer, theta0, batch_size)
data = TrainingSet(events, all_circuits)

# "start" evaluations: initial random theta, before any optimizer step.
if eval_schedule.batch_start or eval_schedule.global_start:
    record_eval(
        "start",
        measure_loss(qmodel, data[:batch_size], trainer.theta, loss, batch_size) if eval_schedule.batch_start else None,
        measure_loss(qmodel, data, trainer.theta, loss, batch_size) if eval_schedule.global_start else None,
    )
    print()

with tqdm(total=n_iter * N_BATCHES_PER_ITER, unit="batch") as bar:
    for n in range(1, n_iter + 1):
        bar.set_description(f"Pass {n:{n_width}d}/{n_iter}")
        reshuffle = cfg.data.reshuffle_between_passes and n > 1
        seed = cfg.seeds.pass_reshuffle + n if reshuffle else None
        for batch in trainer.batches(data, seed):
            last_batch_loss = trainer.step(batch)
            bar.update()
        # batch evaluation: loss of the pass's last batch, before its step;
        # global evaluation: loss on all the events.
        if n in eval_schedule.batch_passes or n in eval_schedule.global_passes:
            record_eval(
                str(n),
                last_batch_loss if n in eval_schedule.batch_passes else None,
                measure_loss(qmodel, data, trainer.theta, loss, batch_size) if n in eval_schedule.global_passes else None,
            )
theta_final = trainer.theta.tolist()
gate_sequence = record_final_circuit(theta_final)
theta_labels = generate_theta_labels(gate_sequence)


# =============================================================================
# OUTPUT GENERATION
# =============================================================================
# Writes the results JSON, the model JSON and a copy of classifier.py to a
# timestamped folder named after the circuit, layers, training settings, and run time.

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
_base_name = (
    f"circuit{circuit_number}_layers{n_layers}_"
    f"{n_iter}iter_batch{batch_size}_{timestamp}"
)
output_dir = f"{cfg.data.run_tag}_{_base_name}" if cfg.data.run_tag else _base_name
os.makedirs(output_dir, exist_ok=True)
print(f"\nSaving outputs to: {output_dir}/")

# --- JSON results file -------------------------------------------------------

json_path = os.path.join(output_dir, "results.json")

results = {
    "run_info": {
        "signal_file": cfg.data.sig_path.name,
        "background_file": cfg.data.bkg_path.name,
        "data_folder": cfg.data.sig_path.resolve().parent.name,
        "timestamp": timestamp,
        "n_signal": len(sig),
        "n_background": len(bkg),
        "n_events_loaded": N_LOADED,
        "n_events_used": N_EVENTS,
        "max_events_limit": max_events,
        "BALANCE_CLASSES": cfg.data.balance_classes,
        "EVENT_ORDER": cfg.data.event_order,
        "DATA_SHUFFLE_SEED": cfg.seeds.data_shuffle,
        "RESHUFFLE_BETWEEN_PASSES": cfg.data.reshuffle_between_passes,
        "PASS_RESHUFFLE_SEED": cfg.seeds.pass_reshuffle,
        "N_ITER": n_iter,
        "BATCH_SIZE_IS_FRACTION": batch_size_is_fraction,
        "BATCH_SIZE": batch_size,
        "N_BATCHES_PER_ITER": N_BATCHES_PER_ITER,
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
        "circuit_number": circuit_number,
        "circuit_description": circuit.description,
        "n_layers": n_layers,
        "total_params": N_PARAMS,
    },
    "output": {
        "theta_final": theta_final,
        "theta_labels": theta_labels,
        "global_eval_checkpoints": global_eval_results,
        "batch_eval_checkpoints": batch_eval_results,
    },
}

with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  [1/3] Results JSON : {json_path}")


# --- Model file and standalone classifier ------------------------------------
# classifier.py rebuilds the circuit from model.json. The examples are training
# events with their final P(signal): running classifier.py alone checks that it
# reproduces them.

_example_idx = [
    *np.flatnonzero(events.labels == 1)[:5],
    *np.flatnonzero(events.labels == 0)[:5],
]
_example_p = qmodel.run([all_circuits[i] for i in _example_idx], theta_final)
model = {
    "timestamp": timestamp,
    "circuit": {
        **cfg.model.circuit.model_dump(mode="json"),
        "description": circuit.description,
    },
    "n_qubits": N_QUBITS,
    "n_feat": N_FEAT,
    "gates": gate_sequence,
    "backend": {
        **cfg.model.backend.model_dump(mode="json"),
        "sim_seed": cfg.seeds.simulator,
    },
    "examples": [
        {
            "features": events.features[i].tolist(),
            "label": int(events.labels[i]),
            "expected_p": float(p),
        }
        for i, p in zip(_example_idx, _example_p)
    ],
}
model_path = os.path.join(output_dir, "model.json")
with open(model_path, "w") as f:
    json.dump(model, f, indent=2)
print(f"  [2/3] Model JSON   : {model_path}")

classifier_path = shutil.copy(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "classifier.py"),
    output_dir,
)
print(f"  [3/3] Classifier   : {classifier_path}")

# --- Final summary -----------------------------------------------------------

print(f"\nTraining complete.")
print(f"  Circuit      : [{circuit_number}] {circuit.description}")
print(f"  Layers       : {n_layers}  |  Parameters: {N_PARAMS}")
print(f"  Events       : {N_EVENTS} used of {N_LOADED} loaded")
print(
    f"  N_iter       : {n_iter}  |  Batches/iter: {N_BATCHES_PER_ITER}  |  Batch size: {batch_size}"
)
print(f"  theta_final  : {np.round(theta_final, 4)}")


def print_last_eval(name, results, ckpts):
    """Print the loss of the last evaluated pass, if there is one."""
    if ckpts:
        n = max(ckpts)
        print(f"  Last {name} eval (n={n}): loss={results[str(n)]:.6f}")


print_last_eval("global", global_eval_results, eval_schedule.global_passes)
print_last_eval("batch ", batch_eval_results, eval_schedule.batch_passes)
print(f"  Output folder: {output_dir}/")

_elapsed = time.time() - _t_start
_sim_info = (
    "statevector (exact)"
    if cfg.model.backend.use_statevector
    else f"Nm={cfg.model.backend.nm} shots"
)
print(f"\n  Elapsed time : {_elapsed:.1f} s  ({_elapsed/60:.1f} min)")
print(
    f"  Per event    : {_elapsed/N_EVENTS:.3f} s/event  "
    f"(N_iter={n_iter}, {_sim_info}, {N_EVENTS} events)"
)
