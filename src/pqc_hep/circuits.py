"""Variational circuits W(theta), numbered as in Figure 2 of Sim et al. (2019).

A circuit is a list of operations applied once (`pre`), followed by a `layer` of
operations repeated n_layers times. Each rotation takes the next theta value,
in order: theta[k] goes to the k-th rotation of the circuit.
"""

from dataclasses import dataclass, field
from enum import Enum

from qiskit import QuantumCircuit


class Gate(Enum):
    # The value is the name of the QuantumCircuit method that applies the gate.
    # Rotations (take one theta value)
    RX = "rx"
    RY = "ry"
    RZ = "rz"
    CRX = "crx"
    CRZ = "crz"
    # Fixed gates
    H = "h"
    CX = "cx"
    CZ = "cz"


ROTATIONS = {Gate.RX, Gate.RY, Gate.RZ, Gate.CRX, Gate.CRZ}


@dataclass
class Operation:
    gate: Gate
    qubits: list[int]  # [qubit] or [control, target]


@dataclass
class Circuit:
    description: str
    layer: list[Operation]  # repeated n_layers times
    # gate not repeated in thelayer, but applied once at the beginning of the circuit
    pre: list[Operation] = field(default_factory=list)

    def operations(self, n_layers):
        return self.pre + self.layer * n_layers

    def rotations(self, n_layers):
        """Gate name of each theta value: rotations(n_layers)[k] uses theta[k]."""
        ops = self.operations(n_layers)
        return [op.gate.value for op in ops if op.gate in ROTATIONS]

    def apply(self, qc: QuantumCircuit, theta, n_layers):
        """Append the circuit to `qc`; theta holds floats or qiskit Parameters."""
        theta = iter(theta)
        for op in self.operations(n_layers):
            gate = getattr(qc, op.gate.value)
            if op.gate in ROTATIONS:
                gate(next(theta), *op.qubits)
            else:
                gate(*op.qubits)
        return qc


# Blocks shared by several circuits.
RX_RZ_common_pattern = [
    Operation(Gate.RX, [0]),
    Operation(Gate.RZ, [0]),
    Operation(Gate.RX, [1]),
    Operation(Gate.RZ, [1]),
    Operation(Gate.RX, [2]),
    Operation(Gate.RZ, [2]),
    Operation(Gate.RX, [3]),
    Operation(Gate.RZ, [3]),
]
RY_RZ_common_pattern = [
    Operation(Gate.RY, [0]),
    Operation(Gate.RZ, [0]),
    Operation(Gate.RY, [1]),
    Operation(Gate.RZ, [1]),
    Operation(Gate.RY, [2]),
    Operation(Gate.RZ, [2]),
    Operation(Gate.RY, [3]),
    Operation(Gate.RZ, [3]),
]
RX_common_pattern = [Operation(Gate.RX, [q]) for q in range(4)]
RY_common_pattern = [Operation(Gate.RY, [q]) for q in range(4)]
RZ_common_pattern = [Operation(Gate.RZ, [q]) for q in range(4)]
H_common_pattern = [Operation(Gate.H, [q]) for q in range(4)]

# Each qubit controls every other one, from q3 down to q0.
ALL_TO_common_pattern = [[3, 2], [3, 1], [3, 0], [2, 3], [2, 1], [2, 0],
              [1, 3], [1, 2], [1, 0], [0, 3], [0, 2], [0, 1]]  # fmt: skip


CIRCUITS = {
    1: Circuit(
        "RX-RZ on each qubit, no entanglement",
        layer=RX_RZ_common_pattern,
    ),
    2: Circuit(
        "RX-RZ on each qubit, CNOT ladder q3->q2->q1->q0",
        layer=RX_RZ_common_pattern
        + [
            Operation(Gate.CX, [3, 2]),
            Operation(Gate.CX, [2, 1]),
            Operation(Gate.CX, [1, 0]),
        ],
    ),
    3: Circuit(
        "RX-RZ on each qubit, CRZ ladder q3->q2->q1->q0",
        layer=RX_RZ_common_pattern
        + [
            Operation(Gate.CRZ, [3, 2]),
            Operation(Gate.CRZ, [2, 1]),
            Operation(Gate.CRZ, [1, 0]),
        ],
    ),
    4: Circuit(
        "RX-RZ on each qubit, CRX ladder q3->q2->q1->q0",
        layer=RX_RZ_common_pattern
        + [
            Operation(Gate.CRX, [3, 2]),
            Operation(Gate.CRX, [2, 1]),
            Operation(Gate.CRX, [1, 0]),
        ],
    ),
    5: Circuit(
        "RX-RZ, CRZ from each qubit to all others, RX-RZ",
        layer=RX_RZ_common_pattern
        + [Operation(Gate.CRZ, pair) for pair in ALL_TO_common_pattern]
        + RX_RZ_common_pattern,
    ),
    6: Circuit(
        "RX-RZ, CRX from each qubit to all others, RX-RZ",
        layer=RX_RZ_common_pattern
        + [Operation(Gate.CRX, pair) for pair in ALL_TO_common_pattern]
        + RX_RZ_common_pattern,
    ),
    7: Circuit(
        "RX-RZ, CRZ (q1->q0, q3->q2), RX-RZ, CRZ (q2->q1)",
        layer=RX_RZ_common_pattern
        + [Operation(Gate.CRZ, [1, 0]), Operation(Gate.CRZ, [3, 2])]
        + RX_RZ_common_pattern
        + [Operation(Gate.CRZ, [2, 1])],
    ),
    8: Circuit(
        "RX-RZ, CRX (q1->q0, q3->q2), RX-RZ, CRX (q2->q1)",
        layer=RX_RZ_common_pattern
        + [Operation(Gate.CRX, [1, 0]), Operation(Gate.CRX, [3, 2])]
        + RX_RZ_common_pattern
        + [Operation(Gate.CRX, [2, 1])],
    ),
    9: Circuit(
        "H on each qubit, CZ ladder q3->q2->q1->q0, RX on each qubit",
        layer=H_common_pattern
        + [
            Operation(Gate.CZ, [3, 2]),
            Operation(Gate.CZ, [2, 1]),
            Operation(Gate.CZ, [1, 0]),
        ]
        + RX_common_pattern,
    ),
    10: Circuit(
        "RY once, then per layer CZ ladder q3->q2->q1->q0 plus CZ(q3,q0), RY",
        pre=RY_common_pattern,
        layer=[
            Operation(Gate.CZ, [3, 2]),
            Operation(Gate.CZ, [2, 1]),
            Operation(Gate.CZ, [1, 0]),
            Operation(Gate.CZ, [3, 0]),
        ]
        + RY_common_pattern,
    ),
    11: Circuit(
        "RY-RZ, CNOT (q1->q0, q3->q2), RY-RZ on q1 and q2, CNOT (q2->q1)",
        layer=RY_RZ_common_pattern
        + [
            Operation(Gate.CX, [1, 0]),
            Operation(Gate.CX, [3, 2]),
            Operation(Gate.RY, [1]),
            Operation(Gate.RZ, [1]),
            Operation(Gate.RY, [2]),
            Operation(Gate.RZ, [2]),
            Operation(Gate.CX, [2, 1]),
        ],
    ),
    12: Circuit(
        "RY-RZ, CZ (q1,q0 and q3,q2), RY-RZ on q1 and q2, CZ (q2,q1)",
        layer=RY_RZ_common_pattern
        + [
            Operation(Gate.CZ, [1, 0]),
            Operation(Gate.CZ, [3, 2]),
            Operation(Gate.RY, [1]),
            Operation(Gate.RZ, [1]),
            Operation(Gate.RY, [2]),
            Operation(Gate.RZ, [2]),
            Operation(Gate.CZ, [2, 1]),
        ],
    ),
    13: Circuit(
        "RY, CRZ ring, RY, CRZ ring in the other direction",
        layer=RY_common_pattern
        + [
            Operation(Gate.CRZ, [3, 0]),
            Operation(Gate.CRZ, [2, 3]),
            Operation(Gate.CRZ, [1, 2]),
            Operation(Gate.CRZ, [0, 1]),
        ]
        + RY_common_pattern
        + [
            Operation(Gate.CRZ, [3, 2]),
            Operation(Gate.CRZ, [0, 3]),
            Operation(Gate.CRZ, [1, 0]),
            Operation(Gate.CRZ, [2, 1]),
        ],
    ),
    14: Circuit(
        "RY, CRX ring, RY, CRX ring in the other direction",
        layer=RY_common_pattern
        + [
            Operation(Gate.CRX, [3, 0]),
            Operation(Gate.CRX, [2, 3]),
            Operation(Gate.CRX, [1, 2]),
            Operation(Gate.CRX, [0, 1]),
        ]
        + RY_common_pattern
        + [
            Operation(Gate.CRX, [3, 2]),
            Operation(Gate.CRX, [0, 3]),
            Operation(Gate.CRX, [1, 0]),
            Operation(Gate.CRX, [2, 1]),
        ],
    ),
    15: Circuit(
        "RY, CNOT ring, RY, CNOT ring in the other direction",
        layer=RY_common_pattern
        + [
            Operation(Gate.CX, [3, 0]),
            Operation(Gate.CX, [2, 3]),
            Operation(Gate.CX, [1, 2]),
            Operation(Gate.CX, [0, 1]),
        ]
        + RY_common_pattern
        + [
            Operation(Gate.CX, [3, 2]),
            Operation(Gate.CX, [0, 3]),
            Operation(Gate.CX, [1, 0]),
            Operation(Gate.CX, [2, 1]),
        ],
    ),
    16: Circuit(
        "RX-RZ, CRZ (q1->q0, q3->q2, q2->q1)",
        layer=RX_RZ_common_pattern
        + [
            Operation(Gate.CRZ, [1, 0]),
            Operation(Gate.CRZ, [3, 2]),
            Operation(Gate.CRZ, [2, 1]),
        ],
    ),
    17: Circuit(
        "RX-RZ, CRX (q1->q0, q3->q2, q2->q1)",
        layer=RX_RZ_common_pattern
        + [
            Operation(Gate.CRX, [1, 0]),
            Operation(Gate.CRX, [3, 2]),
            Operation(Gate.CRX, [2, 1]),
        ],
    ),
    18: Circuit(
        "RX, RZ, CRZ ring",
        layer=RX_common_pattern
        + RZ_common_pattern
        + [
            Operation(Gate.CRZ, [3, 0]),
            Operation(Gate.CRZ, [2, 3]),
            Operation(Gate.CRZ, [1, 2]),
            Operation(Gate.CRZ, [0, 1]),
        ],
    ),
    19: Circuit(
        "RX, RZ, CRX ring",
        layer=RX_common_pattern
        + RZ_common_pattern
        + [
            Operation(Gate.CRX, [3, 0]),
            Operation(Gate.CRX, [2, 3]),
            Operation(Gate.CRX, [1, 2]),
            Operation(Gate.CRX, [0, 1]),
        ],
    ),
}
