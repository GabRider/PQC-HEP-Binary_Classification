"""Training events: loading from the CSV files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# When needed move it into the TOML config file, for now it's a constant in the code.
N_FEAT = 12  # features per event in the CSV files


@dataclass
class Events:
    """One row per event; the three arrays are always indexed together."""

    features: np.ndarray  # shape (n, N_FEAT)
    weights: np.ndarray
    labels: np.ndarray  # 1.0 = signal, 0.0 = background

    @classmethod
    def from_csv(cls, filepath: str | Path, label: int) -> "Events":
        """
        Read one CSV file; every event gets `label` (1 = signal, 0 = background).
        Cols 0-11: float features. Col 12: float weight. Col 13: ignored.
        """
        data = np.loadtxt(filepath, delimiter=",", usecols=range(N_FEAT + 1), ndmin=2)
        return cls(data[:, :N_FEAT], data[:, N_FEAT], np.full(len(data), float(label)))

    def concat(self, other: "Events") -> "Events":
        """New Events: these events followed by those of `other` (self is unchanged)."""
        return Events(
            np.concatenate([self.features, other.features]),
            np.concatenate([self.weights, other.weights]),
            np.concatenate([self.labels, other.labels]),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: np.ndarray | slice) -> "Events":
        """Events selected by an index array, a mask or a slice."""
        return Events(self.features[idx], self.weights[idx], self.labels[idx])
