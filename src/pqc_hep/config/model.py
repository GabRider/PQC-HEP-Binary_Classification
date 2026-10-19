"""[model] table and its sub-tables: circuit, training, adam, checkpoints, backend."""

from typing import Annotated, Literal

from pydantic import Field, PositiveInt, model_validator

from pqc_hep.config.base import Section

Checkpoint = int | Literal["start", "end"]
Fraction = Annotated[float, Field(gt=0, le=1)]


class CircuitConfig(Section):
    circuits_file: str = "circuits_library_Qiskit.py"
    circuit_number: int = 1  # 1-19
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
