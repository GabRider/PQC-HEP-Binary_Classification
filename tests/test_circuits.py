"""circuits.py builds the same gates as Adriano's library (tests/original_circuits.py).

Excluded by default; run it with:
    uv run pytest -m circuit
"""

import original_circuits
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from pqc_hep.circuits import CIRCUITS


def gate_list(qc):
    return [
        (i.operation.name, [qc.find_bit(q).index for q in i.qubits], i.operation.params)
        for i in qc.data
    ]


@pytest.mark.circuit
@pytest.mark.parametrize("number", range(1, 20))
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_same_gates_as_original(number, n_layers):
    build, fixed, per_layer, _ = original_circuits.CIRCUITS[number]
    theta = ParameterVector("theta", fixed + n_layers * per_layer)
    old, old_rotations = build(QuantumCircuit(4), list(theta), n_layers)
    new = CIRCUITS[number].apply(QuantumCircuit(4), theta, n_layers)
    assert gate_list(new) == gate_list(old)
    assert CIRCUITS[number].rotations(n_layers) == old_rotations
