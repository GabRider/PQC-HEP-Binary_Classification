# Page in construction

This fork is the continuation of the original project. The original code is kept
unchanged in [`adriano_code/`](adriano_code/README.md) (references, circuit diagrams).

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and
Python >= 3.11 (uv installs it if needed).

```bash
git clone <repo-url>
cd PQC-HEP-Binary_Classification
uv sync                        # create .venv and install dependencies
uv sync --extra ibm            # optional: real IBM Quantum hardware support
uv run pre-commit install      # optional: enable ruff hooks on commit
```

For IBM hardware, put your token in a `.env` file at the repository root
(ignored by git):

```bash
IBM_TOKEN=your-token
```

## Data

The data files are not published. Place them in `src/pqc_hep/Data/`:
`train_sig.csv` (signal) and `train_bkg.csv` (background).
`val_sig.csv` and `val_bkg.csv` hold the validation events. They are needed
only with a validation set (`phase0.toml`, `uv run pytest -m phase0`).

## Usage

```bash
cd src/pqc_hep
uv run python main.py                          # built-in defaults
uv run python main.py --config my_run.toml     # settings from a TOML file
uv run --env-file ../../.env python main.py --config my_run.toml   # IBM hardware
```

Relative paths in the TOML file are relative to the directory you run from.
Each run writes a folder in `results_dir` (default `result/`), for example:

```
result/c01/2026-10-23/11h20m01_scan_lr_L1_e10_train1k_b100_lr0.005_seed3f9a2c/
```

- `c01`: circuit number; `2026-10-23`: day of the run;
- `11h20m01`: start time, first so that a day's runs sort in order;
- `scan_lr`: name of the `--config` TOML file without `.toml` (absent without
  `--config`). Name the file after what it tests;
- `L1`: layers, `e10`: epochs (`n_iter`), `train1k`: training events,
  `val2k`: validation events (only with a validation set), `b100`: batch size,
  `lr0.005`: learning rate, `seed3f9a2c`: id computed from the seeds (same
  seeds, same id; the seeds themselves are in the TOML copy).

The folder contains:

- `results.json`: final θ and losses, in the format of the original script.
  It is kept only so the tests can compare the results, and gets no new keys.
  The complete configuration is in the TOML copy;
- `model.json`: trained gates, backend settings and 10 training events with
  their final P(signal);
- `<toml>.toml` (`config.toml` without `--config`): the effective configuration,
  including the seeds drawn at random. `--config` on this file reruns the same
  training;
- `classifier.py`: a copy of `src/pqc_hep/classifier.py` that reads `model.json`
  and needs only numpy, qiskit and qiskit-aer;
- `history.csv`: loss and AUC of the training (and validation) events at each
  global evaluation;
- `val_predictions.csv` (with a validation set): label, weight and final
  P(signal) of each validation event.

To classify events, send or open the folder and run:

```bash
python classifier.py              # self-test: reproduces the 10 stored examples
python classifier.py events.csv   # classify every event (columns 0-11: features)
```

### Converting a phase-0 run

A run folder written by the phase-0 script (`PQC_phase0.py`, copied in
`tests/reference/phase0.py`) can be completed into a run folder of `main.py`: `model.json`, the TOML and our `classifier.py`
are written into it, next to its `results.json`, `history.csv` and
`val_predictions.csv`.

```bash
cd src/pqc_hep
uv run python -m pqc_hep.utils.convert_phase0 <phase0_run_folder>              # config.toml
uv run python -m pqc_hep.utils.convert_phase0 <phase0_run_folder> --name scan  # scan.toml
```

Every setting comes from the run's `results.json`. The data paths are relative
(`Data/train_sig.csv`, ...), so the command runs from `src/pqc_hep`. The
converted `model.json` has no stored examples, so `python classifier.py` alone
has nothing to check; classifying a CSV works.

## Configuration

**THIS SECTION IS UNDER DEVELOPMENT AND MAY CHANGE.**

A TOML file has three parts; every key is optional. See
[`src/pqc_hep/example.toml`](src/pqc_hep/example.toml) for all keys and defaults.

| Section               | Content                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `[data]`              | training and validation files, number of events, event order, feature scaling, output folder |
| `[model.circuit]`     | circuits library, circuit number, number of layers                                           |
| `[model.training]`    | number of passes (`n_iter`), batch size                                                      |
| `[model.adam]`        | Adam optimiser parameters                                                                    |
| `[model.evaluation]`  | passes where the loss is measured                                                            |
| `[model.backend]`     | statevector or shots, number of shots, IBM backend                                           |
| `[seeds]`             | the four random seeds                                                                        |

Rules to know:

- A missing seed is drawn at random and saved in the TOML copy of the run
  folder, so every run can be reproduced.
- `batch_size`: an integer is a number of events per batch (`32`), a float is a
  fraction of all events (`0.1`). `1` and `1.0` are not the same.
- A misspelled key, a wrong type or an invalid value stops the program with a
  message naming the field, e.g. `model.adam.alpha: Input should be greater than 0`.

## Refactoring roadmap

Done:

- uv project, ruff and pre-commit setup
- Original code moved, unchanged, to `adriano_code/`
- Equivalence test against the original script (`uv run pytest -m refactoring`)
- `main.py` split into modules (data, quantum, training, output) and passing ruff
- Phase-0 features (z-score, validation set, loss and AUC per evaluation,
  `history.csv`, `val_predictions.csv`, z-score in the classifier), tested against
  the phase-0 script (`uv run pytest -m phase0`)

In progress:

- TOML configuration validated by pydantic (`src/pqc_hep/config/`): implemented,
  awaiting feedback from the colleagues who run `main.py`

To do:

- Cover the statevector mode in the equivalence test
- Remove redundant code and factor out duplicated logic
- Add Sphinx documentation
- Integrate the rest of the phase0 branch: hybrid backend, theta per epoch in
  history.csv, best epoch in results.json, analysis and scan scripts

## Testing the refactoring

```bash
uv run pytest -m refactoring   # training: theta_final and every loss
uv run pytest -m classifier    # classifier.py: predictions on 20 events
uv run pytest -m circuit       # circuits.py: same gates for the 19 circuits
uv run pytest -m phase0        # phase-0 features: theta, history, validation, classifier
```

`refactoring` and `classifier` run `tests/reference/original.py` (a copy of the
original script with fixed seeds) and `main.py` with the same settings. `refactoring` requires identical
training results in the shot-based mode; `classifier` trains in statevector
mode, the only mode where the original classifier is deterministic, and
requires identical predictions. Each test needs the data files (skipped
otherwise), takes about 30 s, and is excluded from a plain `uv run pytest`.

`circuit` compares the gates built by `circuits.py` with those of
`tests/reference/original_circuits.py`. It needs no data.

`phase0` runs `tests/reference/phase0.py` (a copy of the phase-0 script with
fixed seeds) and `main.py` in statevector mode, on six settings. It requires
identical `theta_final` and classifier predictions, and the same `history.csv`
and `val_predictions.csv` within 1e-6 (the phase-0 script rounds them). It takes
about 50 s.

A failure means the refactored code changed the results: a bug, or an intended
change of behaviour that makes this test obsolete.

## Development

[`src/README.md`](src/README.md) describes the role of each source file and
which one to open to modify or add something.

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
```
