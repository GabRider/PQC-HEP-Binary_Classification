"""
check_hybrid_circuits.py

Checks that the 'hybrid' backend of PQC_phase0.py computes the same probabilities as
the 'qiskit' backend for every circuit of the library (the 13 validation runs only
use circuit 6). No training: for each circuit and number of layers, with real events
and several random theta values, p = P(qubit 0 = 1) is computed in both ways:

  - qiskit : StatePreparation + variational circuit, transpiled for AerSimulator,
             parameters bound, simulated gate by gate with Statevector
             (same steps as build_event_circuit / _run_circuits_batch);
  - hybrid : normalised feature vector x, Qiskit Operator U(theta) of the variational
             circuit alone, p = sum over the basis states with q0 = 1 of |U x|^2
             (same steps as _probs_hybrid).

Prints, for each circuit, the largest |p_qiskit - p_hybrid| over all events and theta.
Read-only: no file is created or modified.

Usage (from the repository root):
    python phase0/analysis/check_hybrid_circuits.py
    python phase0/analysis/check_hybrid_circuits.py --layers 1 --n-theta 5 --n-events 25
"""
import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, Operator
from qiskit_aer import AerSimulator

HERE          = Path(__file__).parent                 # phase0/analysis/
REPO          = HERE.parent.parent                    # repository root
CIRCUITS_FILE = REPO / "Qiskit implementation" / "circuits_library_Qiskit.py"
DATA          = REPO / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--layers",   type=int, nargs="+", default=[1, 2], help="numbers of layers to test")
parser.add_argument("--n-theta",  type=int, default=5,  help="random theta values per circuit")
parser.add_argument("--n-events", type=int, default=25, help="events per class (signal, background)")
parser.add_argument("--seed",     type=int, default=1,  help="seed for the random theta values")
args = parser.parse_args()

N_QUBITS, N_DIM, N_FEAT = 4, 16, 12
Q0_ONE_IDX = np.array([k for k in range(N_DIM) if k % 2 == 1])   # basis states with qubit 0 = 1

# --- Circuit library (loaded the same way as in PQC_phase0.py) ---------------
spec = importlib.util.spec_from_file_location("circuits_library", CIRCUITS_FILE)
clib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clib)


# --- Real events: z-score, zero-padding to 16, normalisation -----------------
def load_events(name, n):
    rows = []
    with open(DATA / name) as f:
        for line in f:
            rows.append([float(v) for v in line.split(",")[:N_FEAT]])
            if len(rows) == n:
                break
    return np.array(rows)


X = np.vstack([load_events("train_sig.csv", args.n_events),
               load_events("train_bkg.csv", args.n_events)])
std = X.std(axis=0)
std[std == 0] = 1.0
X = (X - X.mean(axis=0)) / std
X = np.hstack([X, np.zeros((len(X), N_DIM - N_FEAT))])
X /= np.linalg.norm(X, axis=1, keepdims=True)


# --- The two ways of computing p ---------------------------------------------
backend = AerSimulator(method="automatic")


def build_qiskit_circuits(fn, n_params, n_layers):
    """One transpiled circuit per event (encoding + variational form), theta left free."""
    circuits = []
    for x in X:
        pv = ParameterVector("theta", n_params)
        qc = QuantumCircuit(N_QUBITS)
        qc.append(StatePreparation(x.tolist(), normalize=False), range(N_QUBITS))
        full, _ = fn(qc, list(pv), n_layers)
        circuits.append((transpile(full, backend), pv))
    return circuits


def p_qiskit(circuits, theta):
    """Bind theta and simulate each circuit gate by gate (qiskit backend)."""
    return np.array([
        float(Statevector.from_instruction(
            qc.assign_parameters({pv[i]: float(v) for i, v in enumerate(theta)})
        ).probabilities([0])[1])
        for qc, pv in circuits
    ])


def p_hybrid(fn, theta, n_layers):
    """Circuit alone -> U(theta) -> applied to all event vectors at once (hybrid backend)."""
    qc = QuantumCircuit(N_QUBITS)
    qc, _ = fn(qc, [float(v) for v in theta], n_layers)
    U = Operator(qc).data
    psi = X @ U.T
    return np.sum(np.abs(psi[:, Q0_ONE_IDX]) ** 2, axis=1)


# --- Loop over circuits and layers -------------------------------------------
rng = np.random.default_rng(args.seed)
print(f"{len(X)} real events, {args.n_theta} random theta per circuit, layers {args.layers}\n")
print(f"{'circuit':>7} {'L':>2} {'params':>6} | {'max |dp|':>9} | {'qiskit ms/ev':>12} {'hybrid ms/ev':>12}")

worst = 0.0
for number in range(1, len(clib.CIRCUITS)):
    if clib.CIRCUITS[number] is None:
        continue
    fn, fixed_params, layer_params, _ = clib.CIRCUITS[number]
    for n_layers in args.layers:
        n_params = fixed_params + n_layers * layer_params
        circuits = build_qiskit_circuits(fn, n_params, n_layers)
        d_max, t_q, t_h = 0.0, 0.0, 0.0
        for _ in range(args.n_theta):
            theta = rng.uniform(-np.pi, np.pi, n_params)
            t0 = time.perf_counter(); pq = p_qiskit(circuits, theta); t_q += time.perf_counter() - t0
            t0 = time.perf_counter(); ph = p_hybrid(fn, theta, n_layers); t_h += time.perf_counter() - t0
            d_max = max(d_max, float(np.max(np.abs(pq - ph))))
        worst = max(worst, d_max)
        n_eval = args.n_theta * len(X)
        print(f"{number:>7} {n_layers:>2} {n_params:>6} | {d_max:>9.1e} | "
              f"{t_q / n_eval * 1e3:>12.3f} {t_h / n_eval * 1e3:>12.4f}")

print(f"\nLargest difference over all circuits: {worst:.1e}")
print("OK: identical up to machine precision on all circuits." if worst < 1e-10
      else "WARNING: difference above 1e-10.")
