"""Variational circuits W(theta), numbered as in Figure 2 of Sim et al. (2019).

A circuit is a list of operations applied once (`pre`), followed by a `layer` of
operations repeated n_layers times. Each rotation takes the next theta value,
in order: theta[k] goes to the k-th rotation of the circuit.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from qiskit import QuantumCircuit


class ShiftRule(Enum):
    """Exact gradient formula of a gate (parameter-shift rule)."""

    NONE = auto()  # fixed gate, takes no theta
    TWO_TERM = auto()  # [f(θ+π/2) − f(θ−π/2)] / 2
    FOUR_TERM = auto()  # controlled rotations, 4 evaluations


class Gate(Enum):
    # (Qiskit method, parameter-shift rule)
    RX = (QuantumCircuit.rx, ShiftRule.TWO_TERM)
    RY = (QuantumCircuit.ry, ShiftRule.TWO_TERM)
    RZ = (QuantumCircuit.rz, ShiftRule.TWO_TERM)
    CRX = (QuantumCircuit.crx, ShiftRule.FOUR_TERM)
    CRZ = (QuantumCircuit.crz, ShiftRule.FOUR_TERM)
    H = (QuantumCircuit.h, ShiftRule.NONE)
    CX = (QuantumCircuit.cx, ShiftRule.NONE)
    CZ = (QuantumCircuit.cz, ShiftRule.NONE)

    def __call__(self, *qubits):
        """Gate.RX(0) -> Operation(Gate.RX, [0])."""
        return Operation(self, list(qubits))

    @property
    def add_to_circuit(self):
        return self.value[0]

    @property
    def shift_rule(self):
        return self.value[1]

    @property
    def is_rotation(self):
        return self.shift_rule != ShiftRule.NONE


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

    def shift_rules(self, n_layers):
        """Use to compute the gradient of the circuit with respect to theta.

        two_term  : grad[k] = (1/N) * dot(dL_dp, (p+ - p-) / 2)
                    p± evaluated at theta ± π/2.
        four_term : grad[k] = (1/N) * dot(dL_dp, d1*(fa+−fa−) − d2*(fb+−fb−))
                    a-shifts ±π/2, b-shifts ±π, d1=1/2, d2=(√2−1)/4.
        """
        ops = self.operations(n_layers)
        return [op.gate.shift_rule for op in ops if op.gate.is_rotation]

    def apply(self, qc: QuantumCircuit, theta, n_layers):
        """Append the circuit to `qc`; theta holds floats or qiskit Parameters."""
        k = 0  # index of the next theta value
        for op in self.operations(n_layers):
            if op.gate.is_rotation:
                # giving the object through its method is the same as calling qc.x(0)
                # property of a object function see link below
                # https://docs.python.org/3/tutorial/classes.html#method-objects
                op.gate.add_to_circuit(qc, theta[k], *op.qubits)  # = qc.rx(theta[k], q)
                k += 1
            else:
                op.gate.add_to_circuit(qc, *op.qubits)  # = qc.cx(control, target)
        return qc


# Blocks shared by several circuits.
RX_RZ_common_pattern = [
    Gate.RX(0),
    Gate.RZ(0),
    Gate.RX(1),
    Gate.RZ(1),
    Gate.RX(2),
    Gate.RZ(2),
    Gate.RX(3),
    Gate.RZ(3),
]
RY_RZ_common_pattern = [
    Gate.RY(0),
    Gate.RZ(0),
    Gate.RY(1),
    Gate.RZ(1),
    Gate.RY(2),
    Gate.RZ(2),
    Gate.RY(3),
    Gate.RZ(3),
]
RX_common_pattern = [Gate.RX(q) for q in range(4)]
RY_common_pattern = [Gate.RY(q) for q in range(4)]
RZ_common_pattern = [Gate.RZ(q) for q in range(4)]
H_common_pattern = [Gate.H(q) for q in range(4)]

# Each qubit controls every other one, from q3 down to q0.
control_gate_common_pattern = [[3, 2], [3, 1], [3, 0], [2, 3], [2, 1], [2, 0],
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
            Gate.CX(3, 2),
            Gate.CX(2, 1),
            Gate.CX(1, 0),
        ],
    ),
    3: Circuit(
        "RX-RZ on each qubit, CRZ ladder q3->q2->q1->q0",
        layer=RX_RZ_common_pattern
        + [
            Gate.CRZ(3, 2),
            Gate.CRZ(2, 1),
            Gate.CRZ(1, 0),
        ],
    ),
    4: Circuit(
        "RX-RZ on each qubit, CRX ladder q3->q2->q1->q0",
        layer=RX_RZ_common_pattern
        + [
            Gate.CRX(3, 2),
            Gate.CRX(2, 1),
            Gate.CRX(1, 0),
        ],
    ),
    5: Circuit(
        "RX-RZ, CRZ from each qubit to all others, RX-RZ",
        layer=RX_RZ_common_pattern
        + [Gate.CRZ(*pair) for pair in control_gate_common_pattern]
        + RX_RZ_common_pattern,
    ),
    6: Circuit(
        "RX-RZ, CRX from each qubit to all others, RX-RZ",
        layer=RX_RZ_common_pattern
        + [Gate.CRX(*pair) for pair in control_gate_common_pattern]
        + RX_RZ_common_pattern,
    ),
    7: Circuit(
        "RX-RZ, CRZ (q1->q0, q3->q2), RX-RZ, CRZ (q2->q1)",
        layer=RX_RZ_common_pattern
        + [Gate.CRZ(1, 0), Gate.CRZ(3, 2)]
        + RX_RZ_common_pattern
        + [Gate.CRZ(2, 1)],
    ),
    8: Circuit(
        "RX-RZ, CRX (q1->q0, q3->q2), RX-RZ, CRX (q2->q1)",
        layer=RX_RZ_common_pattern
        + [Gate.CRX(1, 0), Gate.CRX(3, 2)]
        + RX_RZ_common_pattern
        + [Gate.CRX(2, 1)],
    ),
    9: Circuit(
        "H on each qubit, CZ ladder q3->q2->q1->q0, RX on each qubit",
        layer=H_common_pattern
        + [
            Gate.CZ(3, 2),
            Gate.CZ(2, 1),
            Gate.CZ(1, 0),
        ]
        + RX_common_pattern,
    ),
    10: Circuit(
        "RY once, then per layer CZ ladder q3->q2->q1->q0 plus CZ(q3,q0), RY",
        pre=RY_common_pattern,
        layer=[
            Gate.CZ(3, 2),
            Gate.CZ(2, 1),
            Gate.CZ(1, 0),
            Gate.CZ(3, 0),
        ]
        + RY_common_pattern,
    ),
    11: Circuit(
        "RY-RZ, CNOT (q1->q0, q3->q2), RY-RZ on q1 and q2, CNOT (q2->q1)",
        layer=RY_RZ_common_pattern
        + [
            Gate.CX(1, 0),
            Gate.CX(3, 2),
            Gate.RY(1),
            Gate.RZ(1),
            Gate.RY(2),
            Gate.RZ(2),
            Gate.CX(2, 1),
        ],
    ),
    12: Circuit(
        "RY-RZ, CZ (q1,q0 and q3,q2), RY-RZ on q1 and q2, CZ (q2,q1)",
        layer=RY_RZ_common_pattern
        + [
            Gate.CZ(1, 0),
            Gate.CZ(3, 2),
            Gate.RY(1),
            Gate.RZ(1),
            Gate.RY(2),
            Gate.RZ(2),
            Gate.CZ(2, 1),
        ],
    ),
    13: Circuit(
        "RY, CRZ ring, RY, CRZ ring in the other direction",
        layer=RY_common_pattern
        + [
            Gate.CRZ(3, 0),
            Gate.CRZ(2, 3),
            Gate.CRZ(1, 2),
            Gate.CRZ(0, 1),
        ]
        + RY_common_pattern
        + [
            Gate.CRZ(3, 2),
            Gate.CRZ(0, 3),
            Gate.CRZ(1, 0),
            Gate.CRZ(2, 1),
        ],
    ),
    14: Circuit(
        "RY, CRX ring, RY, CRX ring in the other direction",
        layer=RY_common_pattern
        + [
            Gate.CRX(3, 0),
            Gate.CRX(2, 3),
            Gate.CRX(1, 2),
            Gate.CRX(0, 1),
        ]
        + RY_common_pattern
        + [
            Gate.CRX(3, 2),
            Gate.CRX(0, 3),
            Gate.CRX(1, 0),
            Gate.CRX(2, 1),
        ],
    ),
    15: Circuit(
        "RY, CNOT ring, RY, CNOT ring in the other direction",
        layer=RY_common_pattern
        + [
            Gate.CX(3, 0),
            Gate.CX(2, 3),
            Gate.CX(1, 2),
            Gate.CX(0, 1),
        ]
        + RY_common_pattern
        + [
            Gate.CX(3, 2),
            Gate.CX(0, 3),
            Gate.CX(1, 0),
            Gate.CX(2, 1),
        ],
    ),
    16: Circuit(
        "RX-RZ, CRZ (q1->q0, q3->q2, q2->q1)",
        layer=RX_RZ_common_pattern
        + [
            Gate.CRZ(1, 0),
            Gate.CRZ(3, 2),
            Gate.CRZ(2, 1),
        ],
    ),
    17: Circuit(
        "RX-RZ, CRX (q1->q0, q3->q2, q2->q1)",
        layer=RX_RZ_common_pattern
        + [
            Gate.CRX(1, 0),
            Gate.CRX(3, 2),
            Gate.CRX(2, 1),
        ],
    ),
    18: Circuit(
        "RX, RZ, CRZ ring",
        layer=RX_common_pattern
        + RZ_common_pattern
        + [
            Gate.CRZ(3, 0),
            Gate.CRZ(2, 3),
            Gate.CRZ(1, 2),
            Gate.CRZ(0, 1),
        ],
    ),
    19: Circuit(
        "RX, RZ, CRX ring",
        layer=RX_common_pattern
        + RZ_common_pattern
        + [
            Gate.CRX(3, 0),
            Gate.CRX(2, 3),
            Gate.CRX(1, 2),
            Gate.CRX(0, 1),
        ],
    ),
}
