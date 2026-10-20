"""Training of theta: training set, evaluation schedule, loss and the Trainer.

The Trainer does not read the configuration: it receives its components (the
QModel, a loss, an optimizer, the initial theta), so each one can be swapped.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from sklearn.metrics import log_loss

from pqc_hep.circuits import ShiftRule
from pqc_hep.config.model import EvaluationConfig
from pqc_hep.data import Events
from pqc_hep.quantum import QModel

# --- What is trained on and when it is evaluated -----------------------------


@dataclass
class TrainingSet:
    """Events and their transpiled circuits; always indexed together."""

    events: Events
    circuits: list[QuantumCircuit]

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: np.ndarray | slice) -> "TrainingSet":
        """Events and circuits selected by an index array or a slice."""
        if isinstance(idx, slice):
            return TrainingSet(self.events[idx], self.circuits[idx])
        return TrainingSet(self.events[idx], [self.circuits[i] for i in idx])


@dataclass
class EvalSchedule:
    """Passes where the losses are measured, and whether "start" is requested.

    The values are checked by the config (ModelConfig._evaluation_fits_n_iter).
    """

    global_passes: set[int]
    global_start: bool
    batch_passes: set[int]
    batch_start: bool

    @classmethod
    def from_config(cls, eval_cfg: EvaluationConfig, n_iter: int) -> "EvalSchedule":
        def passes(raw, step):
            """Pass numbers of one list ("end" = n_iter), filled every `step`."""
            ckpts = {n_iter if v == "end" else v for v in raw if v != "start"}
            if step is not None:
                low = min(ckpts) if len(ckpts) >= 2 else step
                ckpts.update(range(low, max(ckpts) + 1, step))
            return ckpts

        return cls(
            global_passes=passes(eval_cfg.global_at, eval_cfg.global_step),
            global_start="start" in eval_cfg.global_at,
            batch_passes=passes(eval_cfg.batch_at, eval_cfg.batch_step),
            batch_start="start" in eval_cfg.batch_at,
        )


def resolve_batch_size(batch_size: int | float, n_events: int) -> int:
    """Events per batch: an int is kept, a float is a fraction of n_events.
    The batch size must divide n_events, so every batch has the same size."""
    if isinstance(batch_size, float):
        resolved = int(round(batch_size * n_events))
        if resolved < 1:
            raise ValueError(
                f"batch_size={batch_size} resolves to {resolved} events for "
                f"{n_events} total, which is < 1. Use a larger fraction."
            )
        batch_size = resolved
    if n_events % batch_size != 0:
        raise ValueError(f"batch_size={batch_size} does not divide {n_events} events.")
    return batch_size


# --- Loss --------------------------------------------------------------------


class WeightedBCE:
    """Weighted binary cross-entropy, averaged over the events:
    L = (1/N) * sum_i { w_i * [-z_i*ln(p_i) - (1-z_i)*ln(1-p_i)] }
    with p_i = P(q0=1) of event i, z_i its label and w_i its weight.
    """

    def value(self, p: np.ndarray, events: Events) -> float:
        """The loss; sklearn log_loss gives the weighted sum, then divided by N."""
        p_clip = np.clip(p, 1e-7, 1 - 1e-7)
        raw = log_loss(
            events.labels,
            p_clip,
            normalize=False,
            sample_weight=events.weights,
            labels=[0, 1],
        )
        return float(raw / len(p))

    def derivative(self, p: np.ndarray, events: Events) -> np.ndarray:
        """dL_i/dp_i for each event, without the 1/N: how much the event's loss
        changes when its p_i changes a little."""
        p_clip = np.clip(p, 1e-7, 1 - 1e-7)
        z = events.labels
        return events.weights * (-z / p_clip + (1 - z) / (1 - p_clip))


def measure_loss(
    qmodel: QModel, data: TrainingSet, theta: np.ndarray, loss, chunk: int
) -> float:
    """Loss of `data` at the given theta. The circuits run in chunk-sized pieces,
    each piece in a single backend.run() call."""
    p = np.empty(len(data))
    for s in range(0, len(data), chunk):
        e = min(s + chunk, len(data))
        p[s:e] = qmodel.run(data.circuits[s:e], theta)
    return loss.value(p, data.events)


# --- Trainer -----------------------------------------------------------------


class Trainer:
    """Trains theta batch by batch: parameter-shift gradient, then one step of
    the optimizer.

    loss      : object with value(p, events) and derivative(p, events),
                e.g. WeightedBCE.
    optimizer : sklearn-style optimizer, update_params([theta], [grad]) updates
                theta in place (e.g. sklearn AdamOptimizer or SGDOptimizer).
    """

    def __init__(
        self,
        qmodel: QModel,
        loss,
        optimizer,
        theta0: np.ndarray,
        batch_size: int,
    ):
        self.qmodel = qmodel
        self.loss = loss
        self.optimizer = optimizer
        self.theta = theta0
        self.batch_size = batch_size

    def batches(self, data: TrainingSet, seed: int | None) -> Iterator[TrainingSet]:
        """The batches of one pass; the events are reshuffled first when a seed
        is given."""
        if seed is not None:
            data = data[np.random.default_rng(seed=seed).permutation(len(data))]
        for start in range(0, len(data), self.batch_size):
            yield data[start : start + self.batch_size]

    def step(self, batch: TrainingSet) -> float:
        """Gradient of the batch loss, then one optimizer step on theta.
        Returns the batch loss before the step."""
        p = self.qmodel.run(batch.circuits, self.theta)
        dloss_dp = self.loss.derivative(p, batch.events)
        grad = self._parameter_shift(batch.circuits, dloss_dp)
        batch_loss = self.loss.value(p, batch.events)
        self.optimizer.update_params([self.theta], [grad])
        return batch_loss

    def _parameter_shift(
        self, circuits: list[QuantumCircuit], dloss_dp: np.ndarray
    ) -> np.ndarray:
        """Gradient dL/dtheta by the parameter shift rule:
        grad[k] = (1/N) * dot(dL/dp, dp/dtheta[k]), where dp/dtheta[k] comes from
        runs with only theta[k] shifted. Total circuit evaluations:
        (2*n_two + 4*n_four) * batch_size.
        """
        D1 = 0.5
        D2 = (np.sqrt(2) - 1) / 4

        def p_shifted(k, shift):
            """P(q0=1) of the batch with only theta[k] shifted."""
            t = self.theta.copy()
            t[k] += shift
            return self.qmodel.run(circuits, t)

        # Each run is seeded from seed_simulator and the circuit's position in the
        # run, so grouping several shifts into one run would change the shot results.
        grad = np.zeros(len(self.theta))
        for k, rule in enumerate(self.qmodel.shift_rules):
            if rule == ShiftRule.TWO_TERM:
                # RX, RY, RZ: the output is a sine of theta[k] with a single
                # frequency, so two runs at ±π/2 give the exact derivative.
                dp_dtheta = (p_shifted(k, np.pi / 2) - p_shifted(k, -np.pi / 2)) / 2.0
            else:  # FOUR_TERM
                # CRX, CRZ: the rotation only acts when the control is |1>, so
                # the output mixes two frequencies of theta[k]. Two more runs at
                # ±π are needed to separate them.
                diff_a = p_shifted(k, np.pi / 2) - p_shifted(k, -np.pi / 2)
                diff_b = p_shifted(k, np.pi) - p_shifted(k, -np.pi)
                dp_dtheta = D1 * diff_a - D2 * diff_b
            grad[k] = float(np.dot(dloss_dp, dp_dtheta) / len(circuits))
        return grad
