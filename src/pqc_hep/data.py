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


def load_events(sig_path: str | Path, bkg_path: str | Path) -> tuple[Events, Events]:
    """Signal and background events of the two CSV files."""
    sig = Events.from_csv(sig_path, label=1)
    bkg = Events.from_csv(bkg_path, label=0)
    if sig.weights.sum() == 0:
        raise ValueError("signal weight sum is zero.")
    if bkg.weights.sum() == 0:
        raise ValueError("background weight sum is zero.")
    return sig, bkg


def select_events(
    sig: Events,
    bkg: Events,
    max_events: int | None,
    balance: bool,
    rng: np.random.Generator,
) -> Events:
    """Shuffled training events: max_events of them (None = all), with as many
    signal as background events when `balance` is set."""
    if balance:
        n_each = max_events // 2
        available = min(len(sig), len(bkg))
        if n_each > available:
            smaller = "signal" if len(sig) <= len(bkg) else "background"
            raise ValueError(
                f"balance_classes requested {n_each} per class but only "
                f"{available} {smaller} events available. "
                f"Set max_events <= {2 * available} or balance_classes=false."
            )
        s_idx = rng.permutation(len(sig))[:n_each]
        b_idx = rng.permutation(len(bkg))[:n_each]
        selected = sig[s_idx].concat(bkg[b_idx])
        return selected[rng.permutation(len(selected))]

    all_events = sig.concat(bkg)
    if max_events is not None and max_events > len(all_events):
        raise ValueError(
            f"max_events={max_events} exceeds {len(all_events)} available events."
        )
    # [:None] keeps every event
    return all_events[rng.permutation(len(all_events))][:max_events]


def order_events(events: Events, order: str) -> Events:
    """Events in training order: "random" keeps them, "signal_first" or
    "background_first" puts one class before the other."""
    sig, bkg = events[events.labels == 1], events[events.labels == 0]
    if order == "signal_first":
        return sig.concat(bkg)
    if order == "background_first":
        return bkg.concat(sig)
    return events


def normalise_weights(events: Events) -> Events:
    """Events with rescaled weights: sum(w_S) = sum(w_B) = number of background
    events. Done after selection so the balance holds for the trained events."""
    sig_mask, bkg_mask = events.labels == 1, events.labels == 0
    n_bkg = int(bkg_mask.sum())
    weights = events.weights.copy()
    if n_bkg > 0 and sig_mask.sum() > 0:
        weights[sig_mask] *= n_bkg / np.sum(weights[sig_mask])
        weights[bkg_mask] *= n_bkg / np.sum(weights[bkg_mask])
    return Events(events.features, weights, events.labels)


def to_amplitudes(x: np.ndarray, n_dim: int) -> np.ndarray:
    """Amplitudes of one event's quantum state: the features padded with zeros
    to n_dim values, then normalised to unit length."""
    x_padded = np.concatenate([x, np.zeros(n_dim - len(x))])
    norm = np.linalg.norm(x_padded)
    if norm < 1e-12:
        raise ValueError("Feature vector has zero norm and cannot be normalised.")
    return x_padded / norm
