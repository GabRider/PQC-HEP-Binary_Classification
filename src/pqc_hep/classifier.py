"""
Standalone PQC classifier. main.py copies this file next to model.json in each
run folder; it needs only numpy, qiskit and qiskit-aer (plus qiskit-ibm-runtime
for IBM hardware, with the token in the IBM_TOKEN environment variable).

Usage, from the run folder:
    python classifier.py              # self-test on the examples stored in model.json
    python classifier.py events.csv   # classify every event (columns 0-11: features)
"""

import csv
import json
import math
import os
import sys
from functools import cache
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

MODEL = json.loads(Path(__file__).with_name("model.json").read_text())
BACKEND = MODEL["backend"]


def preprocess(features):
    """Pad the features with zeros to 2**n_qubits values, then normalise."""
    x = np.concatenate([features, np.zeros(2 ** MODEL["n_qubits"] - MODEL["n_feat"])])
    norm = np.linalg.norm(x)
    if norm < 1e-12:
        raise ValueError("Feature vector has zero norm and cannot be normalised.")
    return x / norm


def build_circuit(features):
    """Amplitude embedding of the event, then the trained gates of model.json."""
    n_qubits = MODEL["n_qubits"]
    qc = QuantumCircuit(n_qubits)
    amplitudes = preprocess(np.asarray(features, dtype=float))
    qc.append(StatePreparation(amplitudes.tolist(), normalize=False), range(n_qubits))
    for gate in MODEL["gates"]:
        # e.g. {"name": "rx", "qubits": [0], "params": [0.12]} -> qc.rx(0.12, 0)
        getattr(qc, gate["name"])(*gate["params"], *gate["qubits"])
    return qc


@cache
def ibm_device():
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("IBM_TOKEN")
    service = QiskitRuntimeService(token=token) if token else QiskitRuntimeService()
    return service.backend(BACKEND["ibm_backend"])


@cache
def aer_simulator():
    return AerSimulator(
        method=BACKEND["simulator_method"], seed_simulator=BACKEND["sim_seed"]
    )


def probability(qc):
    """P(qubit 0 = |1>), computed with the same method as during training."""
    if BACKEND["use_statevector"]:
        return float(Statevector.from_instruction(qc).probabilities([0])[1])
    qc.measure_all()
    if BACKEND["ibm_backend"] is not None:
        from qiskit_ibm_runtime import SamplerV2

        device = ibm_device()
        sampler = SamplerV2(backend=device)
        job = sampler.run([transpile(qc, device)], shots=BACKEND["nm"])
        counts = job.result()[0].data.meas.get_counts()
    else:
        simulator = aer_simulator()
        job = simulator.run(transpile(qc, simulator), shots=BACKEND["nm"])
        counts = job.result().get_counts()
    # Qiskit little-endian: bitstring is q_{n-1}...q_0, so k[-1] is qubit 0.
    return sum(v for k, v in counts.items() if k[-1] == "1") / sum(counts.values())


def self_test():
    """Classify the training examples of model.json and compare with expected_p."""
    # Shot noise: two estimates of the same p may differ by about 1/sqrt(nm).
    tolerance = 1e-9 if BACKEND["use_statevector"] else 3 / math.sqrt(BACKEND["nm"])
    passed = True
    for i, example in enumerate(MODEL["examples"]):
        p = probability(build_circuit(example["features"]))
        ok = abs(p - example["expected_p"]) <= tolerance
        passed &= ok
        kind = "signal" if example["label"] == 1 else "background"
        print(
            f"Example {i} ({kind:<10}): P(signal) = {p:.4f}  "
            f"expected {example['expected_p']:.4f}  {'OK' if ok else 'MISMATCH'}"
        )
    return passed


def classify_csv(path):
    with open(path, newline="") as f:
        for i, row in enumerate(csv.reader(f)):
            if not row or row[0].startswith("#"):
                continue  # skip blank and comment lines
            p = probability(build_circuit([float(x) for x in row[: MODEL["n_feat"]]]))
            label = "SIGNAL" if p >= 0.5 else "BACKGROUND"
            print(f"Event {i:6d} : P(signal) = {p:.4f}   ->   {label}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        classify_csv(sys.argv[1])
    else:
        sys.exit(0 if self_test() else 1)
