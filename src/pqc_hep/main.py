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
from sklearn.metrics import log_loss
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer

from tqdm import tqdm
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from pqc_hep import circuits
from pqc_hep.circuits import ShiftRule
from pqc_hep.config import load as load_config

# =============================================================================
# CONSTANTS
# Problem dimensions, tied to the data format and the circuit library.
# Fixed by the physics setup; do not change without matching data and circuits.
# =============================================================================
N_QUBITS = 4
N_FEAT = 12
N_DIM = 2**N_QUBITS  # Hilbert-space dimension

_t_start = time.time()


# =============================================================================
# INPUT HANDLING
# =============================================================================
# Loads the TOML run configuration, resolves data file paths, loads the
# circuits library, and prepares evaluation checkpoints.

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

if cfg.data.balance_classes and max_events is None:
    print(
        "Warning: balance_classes=true has no effect when max_events is unset "
        "(all events are used regardless). Set max_events to the desired total "
        "if you want equal signal and background counts."
    )

circuit = circuits.CIRCUITS[circuit_number]
shift_rules = circuit.shift_rules(n_layers)  # one per theta value
N_PARAMS = len(shift_rules)


# Parse evaluation checkpoints.


def parse_checkpoints(raw, n_iter_total, setting_name, auto_step=None):
    """
    Convert a raw checkpoint list into a set of integer N_iter values and a
    flag indicating whether "start" was requested.
    Accepted values: integers in [1, n_iter_total], "start", "end".
    If auto_step is set, fills checkpoints from min to max of the integer set
    at that step interval. When only one integer is in the set (e.g. just "end"),
    filling starts from auto_step up to that single value.
    Returns (checkpoint_int_set, include_start).
    """
    if not raw:
        return set(), False
    ckpt_set = set()
    include_start = False
    for val in raw:
        if val == "start":
            include_start = True
        elif val == "end":
            ckpt_set.add(n_iter_total)
        elif isinstance(val, int):
            if 1 <= val <= n_iter_total:
                ckpt_set.add(val)
            else:
                print(
                    f"Warning: {setting_name} value {val} is outside "
                    f"[1, {n_iter_total}] and will be skipped."
                )
        else:
            print(
                f"Warning: {setting_name} value {repr(val)} is not valid and will be skipped."
            )

    # Auto-fill intermediate checkpoints at regular step intervals.
    # max is taken from the set; min is the set's minimum if 2+ values exist,
    # otherwise auto_step itself (handles the ["start","end"] case where only
    # one integer is present — fills from the first step up to that endpoint).
    if auto_step is not None:
        if len(ckpt_set) == 0:
            print(
                f"Warning: {setting_name}_STEP={auto_step} is set but "
                f"{setting_name} has no integer checkpoints to fill between; "
                f"auto-fill has no effect."
            )
        else:
            before = len(ckpt_set)
            max_val = max(ckpt_set)
            min_val = min(ckpt_set) if len(ckpt_set) >= 2 else auto_step
            for v in range(min_val, max_val + 1, auto_step):
                if 1 <= v <= n_iter_total:
                    ckpt_set.add(v)
            if len(ckpt_set) == before:
                print(
                    f"Warning: {setting_name}_STEP={auto_step} produced no "
                    f"additional checkpoints (step may be larger than the "
                    f"interval between existing checkpoints or N_ITER={n_iter_total})."
                )

    return ckpt_set, include_start


global_ckpt_set, global_do_start = parse_checkpoints(
    cfg.model.checkpoints.global_checkpoints,
    n_iter,
    "GLOBAL_EVAL_CHECKPOINTS",
    auto_step=cfg.model.checkpoints.global_step,
)
batch_ckpt_set, batch_do_start = parse_checkpoints(
    cfg.model.checkpoints.batch_checkpoints,
    n_iter,
    "BATCH_EVAL_CHECKPOINTS",
    auto_step=cfg.model.checkpoints.batch_step,
)


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
  global_eval_at            : start={global_do_start}, n_iter={sorted(global_ckpt_set)}
  batch_eval_at             : start={batch_do_start}, n_iter={sorted(batch_ckpt_set)}
{_rule}
""")


# =============================================================================
# DATA LOADING
# =============================================================================
# Reads signal and background CSV files, selects and orders events, normalises
# weights relative to the selected training set, and validates batch_size.


def load_events(filepath):
    """
    Read one CSV file and return (features, weights).
    Cols 0-11: float features. Col 12: float weight. Col 13: ignored.
    """
    data = np.loadtxt(filepath, delimiter=",", usecols=range(N_FEAT + 1), ndmin=2)
    return data[:, :N_FEAT], data[:, N_FEAT]


print("Loading data...")
s_feats, s_weights = load_events(cfg.data.sig_path)
b_feats, b_weights = load_events(cfg.data.bkg_path)
s_labels = np.ones(len(s_feats))
b_labels = np.zeros(len(b_feats))
if s_weights.sum() == 0:
    print("Error: signal weight sum is zero.")
    sys.exit(1)
if b_weights.sum() == 0:
    print("Error: background weight sum is zero.")
    sys.exit(1)
print(
    f"  Loaded : {len(s_feats) + len(b_feats)} events ({len(s_feats)} signal, {len(b_feats)} background)"
)

# Combine and shuffle all events.
all_feats = np.concatenate([s_feats, b_feats])
all_weights = np.concatenate([s_weights, b_weights])
all_labels = np.concatenate([s_labels, b_labels])
rng = np.random.default_rng(seed=cfg.seeds.data_shuffle)

if cfg.data.balance_classes and max_events is not None:
    n_each = max_events // 2
    available = min(len(s_feats), len(b_feats))
    if n_each > available:
        smaller = "signal" if len(s_feats) <= len(b_feats) else "background"
        print(
            f"Error: BALANCE_CLASSES requested {n_each} per class but only "
            f"{available} {smaller} events available. "
            f"Set MAX_EVENTS <= {2*available} or BALANCE_CLASSES=False."
        )
        sys.exit(1)
    s_idx = rng.permutation(len(s_feats))[:n_each]
    b_idx = rng.permutation(len(b_feats))[:n_each]
    sel_f = np.concatenate([s_feats[s_idx], b_feats[b_idx]])
    sel_w = np.concatenate([s_weights[s_idx], b_weights[b_idx]])
    sel_l = np.concatenate([s_labels[s_idx], b_labels[b_idx]])
    order = rng.permutation(len(sel_f))
    events = sel_f[order]
    weights = sel_w[order]
    labels = sel_l[order]
else:
    if max_events is not None and max_events > len(all_feats):
        print(
            f"Error: MAX_EVENTS={max_events} exceeds {len(all_feats)} available events."
        )
        sys.exit(1)
    order = rng.permutation(len(all_feats))
    events = all_feats[order]
    weights = all_weights[order]
    labels = all_labels[order]
    if max_events is not None:
        events = events[:max_events]
        weights = weights[:max_events]
        labels = labels[:max_events]

if cfg.data.event_order == "signal_first":
    sig_m = labels == 1
    bkg_m = labels == 0
    events = np.concatenate([events[sig_m], events[bkg_m]])
    weights = np.concatenate([weights[sig_m], weights[bkg_m]])
    labels = np.concatenate([labels[sig_m], labels[bkg_m]])
elif cfg.data.event_order == "background_first":
    sig_m = labels == 1
    bkg_m = labels == 0
    events = np.concatenate([events[bkg_m], events[sig_m]])
    weights = np.concatenate([weights[bkg_m], weights[sig_m]])
    labels = np.concatenate([labels[bkg_m], labels[sig_m]])

N_EVENTS = len(events)

# Normalise weights: sum(w_S) = sum(w_B) = N(background events in training set).
# Done after selection so the balance is correct for the events actually trained on.
_sig_mask = labels == 1
_bkg_mask = labels == 0
_n_bkg_train = int(_bkg_mask.sum())
if _n_bkg_train > 0 and _sig_mask.sum() > 0:
    weights[_sig_mask] *= _n_bkg_train / np.sum(weights[_sig_mask])
    weights[_bkg_mask] *= _n_bkg_train / np.sum(weights[_bkg_mask])

print(
    f"  Using  : {N_EVENTS} events ({int(_sig_mask.sum())} signal, {_n_bkg_train} background)"
)
print(f"  Weight normalisation: sum(w_S) = sum(w_B) = {_n_bkg_train}")

# Resolve BATCH_SIZE fraction to an integer now that N_EVENTS is known.
if batch_size_is_fraction:
    _resolved = int(round(float(batch_size) * N_EVENTS))
    if _resolved < 1:
        print(
            f"Error: BATCH_SIZE={batch_size} resolves to {_resolved} events for {N_EVENTS} total, "
            f"which is < 1. Use a larger fraction."
        )
        sys.exit(1)
    batch_size = _resolved
    print(
        f"  Batch size (resolved) : {batch_size}  (= {batch_size/N_EVENTS*100:.1f}% of {N_EVENTS} events)"
    )

# Validate that batch_size evenly divides the event count.
if N_EVENTS % batch_size != 0:
    divisors = [i for i in range(1, N_EVENTS + 1) if N_EVENTS % i == 0]
    print(f"\nError: batch_size={batch_size} does not evenly divide {N_EVENTS} events.")
    if batch_size_is_fraction:
        valid_fracs = [
            f"{d}/{N_EVENTS} = {d/N_EVENTS:.4f}" for d in divisors if 0 < d < N_EVENTS
        ]
        print(f"  Valid batch sizes: {divisors}")
        print(f"  Valid fractions: {valid_fracs[:12]}")
    else:
        print(f"  Valid batch sizes for {N_EVENTS} events: {divisors}")
    sys.exit(1)

N_BATCHES_PER_ITER = N_EVENTS // batch_size
print(f"  Batches per N_iter : {N_BATCHES_PER_ITER}  (batch_size={batch_size})\n")


# =============================================================================
# QUANTUM PIPELINE
# =============================================================================
# Defines all quantum operations: amplitude embedding, transpiled circuit
# building, shot-based or statevector measurement, loss, and gradient via the
# parameter shift rule.

# --- Backend -----------------------------------------------------------------
# Select either the local AerSimulator or a real IBM Quantum computer.

if cfg.model.backend.ibm_backend is not None:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as _IBMSampler
    except ImportError:
        print(
            "Error: model.backend.ibm_backend is set but qiskit_ibm_runtime is not installed."
        )
        print("  Run: pip install qiskit-ibm-runtime")
        sys.exit(1)
    # Token from the IBM_TOKEN environment variable (e.g. uv run --env-file .env);
    # unset = credentials saved in the local qiskit-ibm-runtime account.
    _ibm_token = os.environ.get("IBM_TOKEN")
    if _ibm_token:
        service = QiskitRuntimeService(token=_ibm_token)
    else:
        service = QiskitRuntimeService()
    backend = service.backend(cfg.model.backend.ibm_backend)
    _ibm_sampler = _IBMSampler(backend=backend)
    print(f"  Using real IBM Quantum hardware: {cfg.model.backend.ibm_backend}")
else:
    _ibm_sampler = None
    # max_parallel_experiments=0: use all available CPU cores when a batch of circuits
    # is submitted in a single backend.run() call. Gracefully degrades to sequential
    # execution if only one core is available. No effect in statevector mode.
    backend = AerSimulator(
        method=cfg.model.backend.simulator_method,
        seed_simulator=cfg.seeds.simulator,
        max_parallel_experiments=0,
    )


# --- Preprocessing -----------------------------------------------------------


def preprocess_event(x_raw):
    """
    Pad the 12-dimensional feature vector to length 16 with zeros, then
    normalise to unit length to form a valid quantum-state amplitude vector.
    """
    x_padded = np.concatenate([x_raw, np.zeros(N_DIM - N_FEAT)])
    norm = np.linalg.norm(x_padded)
    if norm < 1e-12:
        raise ValueError("Feature vector has zero norm and cannot be normalised.")
    return x_padded / norm


# --- Amplitude embedding and circuit construction ----------------------------


def embed_state_in_qiskit(amplitudes):
    """Return a QuantumCircuit that prepares the given amplitude vector from |0000>."""
    qc = QuantumCircuit(N_QUBITS)
    qc.append(StatePreparation(amplitudes.tolist(), normalize=False), range(N_QUBITS))
    return qc


def build_event_circuit(amplitudes, theta_params):
    """
    Build and transpile the full circuit for one event: embedding + parameterised
    variational form. Transpiled once and reused across all passes by binding theta values
    via assign_parameters(), avoiding repeated transpilation overhead.
    """
    emb_qc = embed_state_in_qiskit(amplitudes)
    full_qc = circuit.apply(emb_qc, theta_params, n_layers)
    if not cfg.model.backend.use_statevector:
        full_qc.measure_all()
    return transpile(full_qc, backend)


def _run_circuits_batch(batch_circuits, theta):
    """
    Evaluate all circuits in batch_circuits at the given theta and return
    an np.array of P(qubit_0 = |1>) values, one per event.

    Statevector : exact P(|1>) — no backend call, no shot noise.
    IBM hardware: SamplerV2 (qiskit_ibm_runtime modern API), one run() call for the batch.
    Local Aer   : backend.run() for all circuits in a single call so AerSimulator can
                  parallelise across CPU cores (max_parallel_experiments=0).
    """
    bound = [
        qc.assign_parameters({params[i]: float(v) for i, v in enumerate(theta)})
        for qc, params in batch_circuits
    ]
    if cfg.model.backend.use_statevector:
        return np.array(
            [
                float(
                    Statevector.from_instruction(b).probabilities([0])[1]
                )  # [0]=qubit index; [1]=P(qubit 0=|1>)
                for b in bound
            ]
        )
    # Qiskit little-endian: bitstring is q_{n-1}...q_1 q_0, so k[-1] is qubit 0.
    if cfg.model.backend.ibm_backend is not None:
        result = _ibm_sampler.run(bound, shots=cfg.model.backend.nm).result()
        values = []
        for i in range(len(bound)):
            counts = result[i].data.meas.get_counts()
            total = sum(counts.values())
            values.append(
                float(sum(v for k, v in counts.items() if k[-1] == "1") / total)
            )
        return np.array(values)
    job = backend.run(bound, shots=cfg.model.backend.nm)
    result = job.result()
    values = []
    for i in range(len(bound)):
        counts = result.get_counts(i)
        total = sum(counts.values())
        values.append(float(sum(v for k, v in counts.items() if k[-1] == "1") / total))
    return np.array(values)


# --- Loss function -----------------------------------------------------------


def compute_loss(p_values, event_labels, event_weights):
    """
    Weighted binary cross-entropy over a set of events:
      L_K = (1/N) * sum_i { w_i * [-z_i*ln(y_i) - (1-z_i)*ln(1-y_i)] }
    N is the number of events passed in (batch size or full dataset size).
    Uses sklearn log_loss for the weighted sum, then divides by N.
    """
    N = len(p_values)
    p_clip = np.clip(p_values, 1e-7, 1 - 1e-7)
    raw = log_loss(
        event_labels,
        p_clip,
        normalize=False,
        sample_weight=event_weights,
        labels=[0, 1],
    )
    return float(raw / N)



# --- Batch gradient via parameter shift rule ---------------------------------


def compute_batch_gradient_and_loss(
    batch_circuits, batch_labels, batch_weights, theta, shift_rules
):
    """
    Compute the gradient of the batch loss w.r.t. theta and the batch loss value.

    Steps:
      1. Forward pass: run all batch circuits at current theta -> p_current[j].
         The batch loss is computed here for free (no extra circuit runs needed).
      2. Per-event analytical gradient of loss w.r.t. P(|1>):
           dL_j/dp_j = w_j * (-z_j / p_j  +  (1-z_j) / (1-p_j))
      3. Per theta component k the correct exact shift rule is applied:
           two_term  : grad[k] = (1/N) * dot(dL_dp, (p+ - p-) / 2)
                       p± evaluated at theta ± π/2.
           four_term : grad[k] = (1/N) * dot(dL_dp, d1*(fa+−fa−) − d2*(fb+−fb−))
                       a-shifts ±π/2, b-shifts ±π, d1=1/2, d2=(√2−1)/4.
      Total circuit evaluations: (1 + 2*n_two + 4*n_four) * batch_size.

    Returns (gradient_array, batch_loss_scalar).
    batch_loss is the pre-step loss (computed before the Adam update).
    """
    N = len(batch_circuits)
    D1 = 0.5
    D2 = (np.sqrt(2) - 1) / 4

    # Forward pass: all batch events in one backend.run() call.
    p_current = _run_circuits_batch(batch_circuits, theta)
    batch_loss = compute_loss(p_current, batch_labels, batch_weights)

    # Analytical per-event gradient of loss w.r.t. P(|1>).
    p_clip = np.clip(p_current, 1e-7, 1 - 1e-7)
    dL_dp = batch_weights * (-batch_labels / p_clip + (1 - batch_labels) / (1 - p_clip))

    gradients = np.zeros(len(theta))
    for k in range(len(theta)):
        if shift_rules[k] == ShiftRule.TWO_TERM:
            tp = theta.copy()
            tp[k] += np.pi / 2
            tm = theta.copy()
            tm[k] -= np.pi / 2
            p_plus = _run_circuits_batch(batch_circuits, tp)
            p_minus = _run_circuits_batch(batch_circuits, tm)
            dpdtheta = (p_plus - p_minus) / 2.0
        else:  # four_term
            ta_p = theta.copy()
            ta_p[k] += np.pi / 2
            ta_m = theta.copy()
            ta_m[k] -= np.pi / 2
            tb_p = theta.copy()
            tb_p[k] += np.pi
            tb_m = theta.copy()
            tb_m[k] -= np.pi
            f_ap = _run_circuits_batch(batch_circuits, ta_p)
            f_am = _run_circuits_batch(batch_circuits, ta_m)
            f_bp = _run_circuits_batch(batch_circuits, tb_p)
            f_bm = _run_circuits_batch(batch_circuits, tb_m)
            dpdtheta = D1 * (f_ap - f_am) - D2 * (f_bp - f_bm)

        gradients[k] = float(np.dot(dL_dp, dpdtheta) / N)

    return gradients, batch_loss


# --- Global evaluation (all events) ------------------------------------------


def evaluate_global_loss(circuit_list, event_labels, event_weights, theta, desc="Eval"):
    """
    Run all events at the given theta and return the global loss.
    Processes events in batch_size-sized chunks to limit memory usage while
    still submitting each chunk as a single backend.run() call for efficiency.
    """
    n_total = len(circuit_list)
    p_values = np.empty(n_total)
    n_chunks = (n_total + batch_size - 1) // batch_size
    for c in tqdm(range(n_chunks), desc=f"  {desc}", unit="batch", leave=False):
        s = c * batch_size
        e = min(s + batch_size, n_total)
        p_values[s:e] = _run_circuits_batch(circuit_list[s:e], theta)
    return compute_loss(p_values, event_labels, event_weights)


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
# Pre-builds and transpiles one full circuit per event (embedding + variational
# form), initialises theta, and creates the sklearn Adam optimiser.

# Pre-build and transpile one full circuit per event (embedding + variational form).
# All N_iter passes reuse these transpiled circuits by binding parameter values.
print(f"Pre-building and transpiling circuits ({N_EVENTS} events)...")
all_circuits = []
for x_raw in tqdm(events, desc="Building", unit="event", leave=True):
    amplitudes = preprocess_event(x_raw)
    theta_params = ParameterVector("theta", N_PARAMS)
    all_circuits.append((build_event_circuit(amplitudes, theta_params), theta_params))
print()

_n_two = shift_rules.count(ShiftRule.TWO_TERM)
_n_four = shift_rules.count(ShiftRule.FOUR_TERM)
print(
    f"  Shift rule classification: {_n_two} two-term (Rx/Ry/Rz), "
    f"{_n_four} four-term (CRx/CRy/CRz)\n"
)

# Initialise theta uniformly in [-pi, pi].
rng_theta = np.random.default_rng(seed=cfg.seeds.theta_init)
theta_current = rng_theta.uniform(-np.pi, np.pi, N_PARAMS)

# Initialise sklearn AdamOptimizer.
# params is a list containing theta_current; update_params() modifies it in place.
optimizer = AdamOptimizer(
    params=[theta_current],
    learning_rate_init=cfg.model.adam.alpha,
    beta_1=cfg.model.adam.beta1,
    beta_2=cfg.model.adam.beta2,
    epsilon=cfg.model.adam.epsilon,
)

# Storage for checkpoint results.
global_eval_results = {}  # key: "start" or str(n_iter) -> loss value (float)
batch_eval_results = {}  # key: "start" or str(n_iter) -> loss value (float)


n_width = len(str(n_iter))
label_width = max(
    5, n_width
)  # "start" is 5 chars; ensures [n=start] aligns with [n=  N]

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
# PRE-TRAINING EVALUATIONS  ("start" checkpoints, before any Adam step)
# =============================================================================

if global_do_start or batch_do_start:
    _b_start = "--------"
    _g_start = "--------"
    if global_do_start:
        loss = evaluate_global_loss(
            all_circuits, labels, weights, theta_current, desc="  Global [start]"
        )
        global_eval_results["start"] = float(loss)
        _g_start = f"{loss:.6f}"
    if batch_do_start:
        # Evaluate the first batch with the initial random theta before any Adam step.
        p_start = _run_circuits_batch(all_circuits[:batch_size], theta_current)
        loss_start = compute_loss(p_start, labels[:batch_size], weights[:batch_size])
        batch_eval_results["start"] = float(loss_start)
        _b_start = f"{loss_start:.6f}"
    print(f"  [n={'start':>{label_width}}]  batch = {_b_start}   GLOBAL = {_g_start}")
    print()


# =============================================================================
# TRAINING LOOP
# =============================================================================
# Runs N_iter passes over all events. Each pass sweeps through N_BATCHES_PER_ITER
# mini-batches, computing one batch gradient and taking one Adam step per batch.

last_batch_loss = None

pass_bar = tqdm(total=n_iter, desc="Training", unit="pass", position=0, leave=True)

for current_iter in range(1, n_iter + 1):

    # Optionally reshuffle the event order at the start of each pass after the first.
    if cfg.data.reshuffle_between_passes and current_iter > 1:
        rng_reshuffle = np.random.default_rng(
            seed=cfg.seeds.pass_reshuffle + current_iter
        )
        perm = rng_reshuffle.permutation(N_EVENTS)
        pass_circuits = [all_circuits[i] for i in perm]
        pass_labels = labels[perm]
        pass_weights = weights[perm]
    else:
        pass_circuits = all_circuits
        pass_labels = labels
        pass_weights = weights

    # Sweep through all batches for this pass, with a per-batch progress bar.
    with tqdm(
        range(N_BATCHES_PER_ITER),
        desc=f"  Pass {current_iter:{n_width}d}/{n_iter}",
        unit="batch",
        leave=False,
        position=1,
    ) as batch_bar:
        for batch_idx in batch_bar:
            start = batch_idx * batch_size
            end = start + batch_size

            b_circuits = pass_circuits[start:end]
            b_labels = pass_labels[start:end]
            b_weights = pass_weights[start:end]

            # Compute gradient and pre-step batch loss via parameter shift rule.
            grad, current_batch_loss = compute_batch_gradient_and_loss(
                b_circuits, b_labels, b_weights, theta_current, shift_rules
            )

            # One Adam step: sklearn AdamOptimizer updates theta_current in place.
            optimizer.update_params([theta_current], [grad])

            # Track the last batch loss of this pass for BATCH_EVAL_CHECKPOINTS.
            last_batch_loss = current_batch_loss

    pass_bar.update(1)

    # Eval checkpoints: batch (last batch pre-step loss) and/or global (all events).
    # Both are shown on the same line for easy comparison.
    if current_iter in batch_ckpt_set or current_iter in global_ckpt_set:
        _b_str = "--------"
        _g_str = "--------"
        if current_iter in batch_ckpt_set:
            batch_eval_results[str(current_iter)] = float(last_batch_loss)
            _b_str = f"{last_batch_loss:.6f}"
        if current_iter in global_ckpt_set:
            g_loss = evaluate_global_loss(
                all_circuits,
                labels,
                weights,
                theta_current,
                desc=f"  Global [n={current_iter:{n_width}d}]",
            )
            global_eval_results[str(current_iter)] = float(g_loss)
            _g_str = f"{g_loss:.6f}"
        tqdm.write(
            f"  [n={current_iter:{label_width}d}]  batch = {_b_str}   GLOBAL = {_g_str}"
        )

pass_bar.close()
theta_final = theta_current.tolist()
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
        "n_signal": len(s_feats),
        "n_background": len(b_feats),
        "n_events_loaded": len(all_feats),
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
        "GLOBAL_EVAL_CHECKPOINTS": sorted(global_ckpt_set),
        "GLOBAL_EVAL_STEP": cfg.model.checkpoints.global_step,
        "BATCH_EVAL_CHECKPOINTS": sorted(batch_ckpt_set),
        "BATCH_EVAL_STEP": cfg.model.checkpoints.batch_step,
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

_example_idx = [*np.flatnonzero(labels == 1)[:5], *np.flatnonzero(labels == 0)[:5]]
_example_p = _run_circuits_batch([all_circuits[i] for i in _example_idx], theta_final)
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
            "features": events[i].tolist(),
            "label": int(labels[i]),
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
print(f"  Events       : {N_EVENTS} used of {len(all_feats)} loaded")
print(
    f"  N_iter       : {n_iter}  |  Batches/iter: {N_BATCHES_PER_ITER}  |  Batch size: {batch_size}"
)
print(f"  theta_final  : {np.round(theta_final, 4)}")
if global_eval_results:
    last_g = sorted((k for k in global_eval_results if k != "start"), key=int)
    if last_g:
        print(
            f"  Last global eval (n={last_g[-1]}): loss={global_eval_results[last_g[-1]]:.6f}"
        )
if batch_eval_results:
    last_b = sorted((k for k in batch_eval_results if k != "start"), key=int)
    if last_b:
        print(
            f"  Last batch  eval (n={last_b[-1]}): loss={batch_eval_results[last_b[-1]]:.6f}"
        )
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
