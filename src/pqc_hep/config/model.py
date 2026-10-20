"""[model] table and its sub-tables: circuit, training, adam, checkpoints, backend."""

from typing import Annotated, Literal

from pydantic import Field, PositiveInt, model_validator

from pqc_hep.config.base import Section

Checkpoint = PositiveInt | Literal["start", "end"]
Fraction = Annotated[float, Field(gt=0, le=1)]


def _check_checkpoints(name, values, step, n_iter):
    if any(isinstance(v, int) and v > n_iter for v in values):
        raise ValueError(f"{name}_checkpoints values must be <= n_iter={n_iter}.")
    if step is not None and set(values) <= {"start"}:
        raise ValueError(f"{name}_step needs a number or 'end' in {name}_checkpoints.")


class CircuitConfig(Section):
    circuit_number: int = Field(1, ge=1, le=19)
    n_layers: PositiveInt = 1


class TrainingConfig(Section):
    n_iter: PositiveInt = 10  # epochs
    batch_size: PositiveInt | Fraction = 0.1  # int = events per batch, float = fraction


class AdamConfig(Section):
    alpha: float = Field(5e-3, gt=0)  # learning rate
    beta1: float = Field(0.9, ge=0, lt=1)
    beta2: float = Field(0.999, ge=0, lt=1)
    epsilon: float = Field(1e-8, gt=0)


class CheckpointsConfig(Section):
    global_checkpoints: list[Checkpoint] = ["start", "end"]
    batch_checkpoints: list[Checkpoint] = ["start", "end"]
    global_step: PositiveInt | None = None  # auto-fill interval
    batch_step: PositiveInt | None = None


class BackendConfig(Section):
    use_statevector: bool = False  # exact P(|1>), no shot noise
    nm: PositiveInt = 1000  # shots per circuit
    simulator_method: Literal[
        "automatic", "statevector", "matrix_product_state", "density_matrix"
    ] = "automatic"
    ibm_backend: str | None = None  # IBM QPU name; token from IBM_TOKEN env variable

    @model_validator(mode="after")
    def _ibm_needs_shots(self):
        if self.ibm_backend is not None and self.use_statevector:
            raise ValueError("ibm_backend requires use_statevector=false.")
        return self


class ModelConfig(Section):
    circuit: CircuitConfig = CircuitConfig()
    training: TrainingConfig = TrainingConfig()
    adam: AdamConfig = AdamConfig()
    checkpoints: CheckpointsConfig = CheckpointsConfig()
    backend: BackendConfig = BackendConfig()

    @model_validator(mode="after")
    def _checkpoints_fit_n_iter(self):
        """Check the evaluation checkpoints against n_iter.

        A checkpoint is a training pass where the loss is measured, printed and
        saved in results.json. There are two kinds of measure:
        - global: loss on all the events with the current theta. Exact, but every
          circuit runs again.
        - batch: loss of the last batch of the pass, already computed by the
          training. Free, but noisy.

        Each list takes pass numbers, "start" (before the first Adam step, with the
        random initial theta) and "end" (= n_iter). The step fills the passes
        between the checkpoints at a regular interval. Example with n_iter = 50:
            ["start", "end"]         -> {50}, plus start
            [10, 20, "end"]          -> {10, 20, 50}
            ["end"], step = 10       -> {10, 20, 30, 40, 50}
            [10, 30], step = 5       -> {10, 15, 20, 25, 30}

        A pass number above n_iter is never reached, and a step needs a pass
        number or "end" to fill up to: both are errors.
        """
        ckpt, n_iter = self.checkpoints, self.training.n_iter
        _check_checkpoints("global", ckpt.global_checkpoints, ckpt.global_step, n_iter)
        _check_checkpoints("batch", ckpt.batch_checkpoints, ckpt.batch_step, n_iter)
        return self
