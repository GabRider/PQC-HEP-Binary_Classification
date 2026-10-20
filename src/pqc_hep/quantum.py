"""Quantum part of the classifier: the event circuits and the backend that runs them."""

import os

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from pqc_hep.circuits import Circuit
from pqc_hep.config.model import BackendConfig
from pqc_hep.data import to_amplitudes

# Fixed by the physics setup; do not change without matching data and circuits.
N_QUBITS = 4
N_DIM = 2**N_QUBITS  # Hilbert-space dimension


def make_backend(backend_cfg: BackendConfig, seed: int):
    """(backend, sampler): a real IBM Quantum computer with its sampler, or the
    local AerSimulator with no sampler."""
    if backend_cfg.ibm_backend is None:
        # max_parallel_experiments=0: use all available CPU cores when a batch of
        # circuits is submitted in a single backend.run() call. No effect in
        # statevector mode.
        backend = AerSimulator(
            method=backend_cfg.simulator_method,
            seed_simulator=seed,
            max_parallel_experiments=0,
        )
        return backend, None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as e:
        raise ImportError(
            "model.backend.ibm_backend is set but qiskit_ibm_runtime is not "
            "installed. Run: pip install qiskit-ibm-runtime"
        ) from e
    # Token from the IBM_TOKEN environment variable (e.g. uv run --env-file .env);
    # unset = credentials saved in the local qiskit-ibm-runtime account.
    service = QiskitRuntimeService(token=os.environ.get("IBM_TOKEN"))
    backend = service.backend(backend_cfg.ibm_backend)
    return backend, SamplerV2(backend=backend)


def prob_qubit0_is_1(counts: dict[str, int]) -> float:
    """Estimate P(qubit 0 = |1>) from the shots of one circuit.

    `counts` maps each measured bitstring to its number of shots, for example
    {"0101": 37, "1100": 12}. Qiskit writes qubit 0 as the last character, so
    the estimate is the share of shots whose bitstring ends with "1".
    """
    return sum(v for k, v in counts.items() if k[-1] == "1") / sum(counts.values())


class QModel:
    """Variational circuit W(theta) of the run and the backend that executes it.

    It does not read the configuration: it receives its components, so each
    one can be swapped.

    backend    : AerSimulator or IBM backend (see make_backend).
    shots      : shots per circuit; None = exact statevector, no measurement.
    sampler    : SamplerV2 for an IBM backend, None otherwise.
    preprocess : preprocess(x, n_dim) turns one event's features into n_dim
                 amplitudes of unit norm (default: zero padding + L2 norm).
    """

    def __init__(
        self,
        circuit: Circuit,
        n_layers: int,
        backend,
        shots: int | None,
        sampler=None,
        preprocess=to_amplitudes,
    ):
        self.circuit = circuit
        self.n_layers = n_layers
        self.shift_rules = circuit.shift_rules(n_layers)  # one per theta value
        # Shared by every event circuit; run() binds the theta values to it.
        self.theta = ParameterVector("theta", len(self.shift_rules))
        self.backend = backend
        self.shots = shots
        self.sampler = sampler
        self.preprocess = preprocess

    def build(self, features: np.ndarray) -> list[QuantumCircuit]:
        """One transpiled circuit per event: embedding + W(theta) (+ measurement).

        Transpiled once; every evaluation then only binds the theta values. One
        transpile() call for all events: Qiskit spreads them over the CPU cores.
        """
        circuits = []
        for x in features:
            qc = QuantumCircuit(N_QUBITS)
            # Amplitude embedding: prepare the event's 16 amplitudes from |0000>.
            amplitudes = self.preprocess(x, N_DIM)
            qc.append(
                StatePreparation(amplitudes.tolist(), normalize=False), range(N_QUBITS)
            )
            self.circuit.apply(qc, self.theta, self.n_layers)
            if self.shots is not None:
                qc.measure_all()
            circuits.append(qc)
        return transpile(circuits, self.backend)

    def run(self, circuits: list[QuantumCircuit], theta: np.ndarray) -> np.ndarray:
        """P(qubit 0 = |1>) of each circuit at the given theta values.

        Statevector : exact P(|1>), no backend call, no shot noise.
        IBM hardware: SamplerV2, one run() call for all the circuits.
        Local Aer   : backend.run() for all the circuits in a single call, so
                      AerSimulator can spread them over the CPU cores.
        """
        bound = [qc.assign_parameters({self.theta: theta}) for qc in circuits]
        if self.shots is None:
            # Exact mode: compute the final state of each circuit instead of
            # measuring it. probabilities([0]) gives [P(q0=0), P(q0=1)] and [1]
            # keeps P(q0=1).
            return np.array([Statevector(b).probabilities([0])[1] for b in bound])
        # Shot mode: each circuit runs `shots` times and the outcomes are counted.
        if self.sampler is not None:
            result = self.sampler.run(bound, shots=self.shots).result()
            counts = [r.data.meas.get_counts() for r in result]
        else:
            result = self.backend.run(bound, shots=self.shots).result()
            counts = [result.get_counts(i) for i in range(len(bound))]
        return np.array([prob_qubit0_is_1(c) for c in counts])
