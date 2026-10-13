# Phase 0 – learning rate, batch size and feature standardisation

Study of the training settings on circuit 6, L = 1, before comparing circuits.
The training code is shared by all phases and lives in `Qiskit implementation/`;
this folder contains everything specific to Phase 0.

```
phase0/
├── README.md          this file
├── scans/             shell scripts that launch the runs (several in parallel)
├── results/           one folder per run (history.csv, val_predictions.csv, results.json)
├── analysis/          scripts that read results/ and make the figures
└── figures/           figures of Phase 0 (PNG and PDF)
```

## Training script: `Qiskit implementation/PQC_phase0.py`

Copy of `PQC_HEP_training_file_Qiskit.py` (unchanged) with:

- **Standardisation** (`FEATURE_SCALING = "zscore"`, or `--scaling none`): each feature is
  standardised with the mean and standard deviation of the *training* events only.
  Without it, the four GeV-scale variables carry ~99 % of the squared amplitudes after
  amplitude encoding (see `analysis/feature_amplitude_shares.py`).
- **Validation sample**: `N_VAL = 2000` events drawn at random (fixed seed) from
  `val_sig.csv` / `val_bkg.csv`, standardised with the training mean/std, weights
  normalised as for training. Used only for monitoring, never for gradients.
  The final train/validation/test split is left for a later phase.
- **Monitoring**: loss and weighted ROC AUC on training and validation at epoch 0 and
  after every epoch.
- **New outputs** in each run folder:
  - `history.csv`: `epoch, step, train_loss, train_auc, val_loss, val_auc`
  - `val_predictions.csv`: `label, weight, p` on the validation sample at the end
  - `results.json` also stores the scaling mode and the training mean/std
- **New options**: `--alpha`, `--scaling {zscore,none}`, `--n-val`, `--run-tag`.
  The learning rate is added to the run folder name; runs with `--scaling none`
  are prefixed with `raw_`.
- **Reproducible defaults**: `USE_STATEVECTOR = True`, `DATA_SHUFFLE_SEED = 1`,
  `THETA_INIT_SEED = 1`.
- **Standalone `classifier.py`**: the training mean/std are written into it, so it takes
  raw features (as in the CSV files) and applies the standardisation itself.

## Usage

The data files are expected in `data/` at the repository root.

```bash
# scans (runs in parallel; run folders go to phase0/results/, logs to phase0/logs/)
caffeinate -i bash phase0/scans/scan_alpha.sh
caffeinate -i bash phase0/scans/scan_batch.sh
caffeinate -i bash phase0/scans/scan_raw.sh

# figures (read phase0/results/, write phase0/figures/)
python phase0/analysis/plot_phase0.py

# a single run, from "Qiskit implementation/"
python PQC_phase0.py --circuit-number 6 --n-layers 1 --data-folder ../data \
    --max-events 1000 --n-iter 50 --batch-size 100 --alpha 0.05
```

`caffeinate -i` keeps a Mac awake during long runs (macOS only).

## Settings and results

Common settings: circuit 6, L = 1, 1000 training events, 2000 validation events,
50 epochs, seeds = 1.

| Scan | Runs | Best (final validation AUC) |
|---|---|---|
| Learning rate (b = 100, z-score) | α = 0.001, 0.005, 0.01, 0.05, 0.1 | α = 0.05 (0.701) |
| Batch size (α = 0.05, z-score) | b = 10, 100, 1000 | b = 100 (0.701) |
| Standardisation (b = 100) | raw features, α = 0.005 and 0.05 | raw: 0.597 and 0.611 vs. z-score: 0.685 and 0.701 |

Retained settings: z-score, α = 0.05, b = 100 events, 50 epochs.
Statistical uncertainty on the AUC with 2000 validation events: about 0.01.
