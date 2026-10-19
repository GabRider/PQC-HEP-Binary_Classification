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

## Usage

```bash
cd src/pqc_hep
uv run python main.py                          # built-in defaults
uv run python main.py --config my_run.toml     # settings from a TOML file
uv run --env-file ../../.env python main.py --config my_run.toml   # IBM hardware
```

Relative paths in the TOML file are relative to the directory you run from.
Each run writes a folder `<run_tag>_circuit<N>_layers<L>_<iter>iter_batch<B>_<timestamp>/`
in the current directory, containing `results.json` (full training record) and
`classifier.py` (standalone classifier with the trained parameters).

## Configuration

**THIS SECTION IS UNDER DEVELOPMENT AND MAY CHANGE.**

A TOML file has three parts; every key is optional. See
[`src/pqc_hep/example.toml`](src/pqc_hep/example.toml) for all keys and defaults.

| Section               | Content                                                       |
| --------------------- | ------------------------------------------------------------- |
| `[data]`              | input files, number of events, event order, output folder tag |
| `[model.circuit]`     | circuits library, circuit number, number of layers            |
| `[model.training]`    | number of passes (`n_iter`), batch size                       |
| `[model.adam]`        | Adam optimiser parameters                                     |
| `[model.checkpoints]` | when the loss is recorded                                     |
| `[model.backend]`     | statevector or shots, number of shots, IBM backend            |
| `[seeds]`             | the four random seeds                                         |

Rules to know:

- A missing seed is drawn at random and recorded in `results.json`, so every run
  can be reproduced.
- `batch_size`: an integer is a number of events per batch (`32`), a float is a
  fraction of all events (`0.1`). `1` and `1.0` are not the same.
- A misspelled key, a wrong type or an invalid value stops the program with a
  message naming the field, e.g. `model.adam.alpha: Input should be greater than 0`.

## Refactoring roadmap

Done:

- uv project, ruff and pre-commit setup
- Original code moved, unchanged, to `adriano_code/`
- Equivalence test against the original script (`uv run pytest -m refactoring`)

In progress:

- TOML configuration validated by pydantic (`src/pqc_hep/config/`), replacing the
  USER SETTINGS block and the command-line options

To do:

- Split `main.py` into modules (data loading, circuits, training, outputs)
- Load the circuits library as a normal import instead of a file path
- Make `main.py` and `circuits_library_Qiskit.py` pass ruff, then remove them from
  the ruff excludes
- Cover the statevector mode in the equivalence test
- Remove redundant code and factor out duplicated logic
- Add Sphinx documentation
- Integrate the features developed on the other branches

## Testing the refactoring

```bash
uv run pytest -m refactoring
```

This test runs `tests/original.py` (a copy of the original script with fixed
seeds) and `main.py` with the same settings, and requires identical results
(`theta_final` and every loss). It needs the data files (skipped otherwise),
takes about 30 s and covers the shot-based mode. It is excluded from a plain
`uv run pytest`.

A failure means the refactored code changed the results: a bug, or an intended
change of behaviour that makes this test obsolete.

## Development

```bash
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest              # tests (without the refactoring test)
```

`main.py` and `circuits_library_Qiskit.py` still come from the original code and
do not pass ruff yet, so they are excluded from ruff (`extend-exclude` in
`pyproject.toml`). Remove them from that list once they are cleaned up.
