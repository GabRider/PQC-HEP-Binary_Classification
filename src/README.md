# Source code of the PQC classifier

This folder contains the program that trains a quantum circuit to distinguish
signal events from background events. Each event is described
by 12 features, encoded in the state of 4 qubits. A parameterized quantum
circuit (PQC) W(θ) is then applied, and the probability of measuring qubit 0 in
|1> is read as P(signal). Training adjusts θ so that this probability is close
to 1 for signal events and close to 0 for background events.

This README tells where each part of the process lives, so you know which file
to open to modify or add something. `main.py` runs the whole process: it reads
a configuration, loads the events, trains θ, and saves the trained model in a
run folder.

```
src
├── pqc_hep
│   ├── config/        # reads and validates the TOML run configuration
│   ├── example.toml   # example configuration, every key with its default
│   ├── main.py        # runs the training from start to end
│   ├── data.py        # loads and prepares the events
│   ├── circuits.py    # the 19 variational circuits W(θ)
│   ├── quantum.py     # builds and runs the circuit of each event
│   ├── training.py    # loss, gradient and optimization of θ
│   ├── output.py      # results files and final summary
│   ├── classifier.py  # standalone classifier copied into each run folder
│   └── Data/          # CSV event files (not versioned)
└── README.md
```

The run folders, named `circuit<N>_layers<L>_<n_iter>iter_batch<B>_<timestamp>/`,
are created in the current directory and are not versioned. After a change,
`uv run pytest -m refactoring` checks that training still gives the same
results as the original code (see the root `Readme.md`).

## `config/` and `example.toml`

A run is described by one TOML file: `python main.py --config my_run.toml`.
Every key is optional, and `example.toml` lists them with their default values.
`config/` reads this file and turns it into a `Config` object that the rest of
the code uses. Each TOML table has its own file: `data.py` for `[data]` (input
files, event selection), `model.py` for `[model]` and its sub-tables (circuit,
training, Adam, evaluation, backend), and `seeds.py` for `[seeds]`. `config.py`
assembles them and provides `load()`.

The configuration is checked when it is loaded, by the rules of `base.py`: an
unknown key, a wrong type or an invalid value stops the program with a message
naming the field, for example `model.adam.alpha: Input should be greater than 0`.
Hence a mistake in the file shows up before training starts. A seed missing
from the file is drawn at random and recorded in `results.json`, so any run can
be repeated.

Where to change:

- New setting: add a field with its type and default value to the class of its
  table (e.g. `TrainingConfig` in `config/model.py`), then add the key to
  `example.toml`. The code reads it as `cfg.model.training.<key>`.
- New table: a new class that inherits from `Section`, added as a field of its
  parent table.
- Constraint on a value (range, allowed strings): in the field's type, e.g.
  `Field(1, ge=1, le=19)`.

## `main.py`

`main.py` contains no computation of its own. It calls the other files in the
order of the run: configuration, data, circuits, training, outputs. It also
prints the effective configuration at the start, a progress bar during
training, and the losses at the passes requested in the configuration.

Where to change:

- Order of the steps, or what is printed during the run.
- Choice of the loss or of the optimizer: both are created in the TRAINING LOOP
  section and given to the `Trainer`.

## `data.py`

The events come from two CSV files, one for signal and one for background.
`data.py` reads them, selects `max_events` events at random (with as many signal
as background events if `balance_classes` is set), and puts them in the
training order chosen in the configuration. The event weights are then rescaled
so that signal and background have the same total weight, which keeps the more
numerous class from dominating the loss. Last, `to_amplitudes()` prepares the
quantum encoding: the 12 features are padded with zeros to 16 values (the
2^4 amplitudes of 4 qubits) and normalized to unit length.

Where to change:

- Event selection or ordering: `select_events()`, `order_events()`.
- Weighting: `normalise_weights()`.
- Encoding of the features: `to_amplitudes()`. `classifier.py` has its own copy
  of this function (`preprocess()`), which must be changed the same way.

## `circuits.py`

This file defines the 19 variational circuits W(θ), numbered as in Figure 2 of
Sim et al. (2019). Each circuit is a sequence of rotations (RX, RY, RZ, and
controlled rotations CRX, CRZ) and entangling gates (CNOT, CZ). The `pre` part
is applied once, then the `layer` part is repeated `n_layers` times, and each
rotation takes the next value of θ. Each gate also records its parameter shift
rule, which `training.py` needs to compute the gradient.

Where to change:

- New circuit: add an entry to `CIRCUITS`, then raise the maximum of
  `circuit_number` in `config/model.py` (`le=19`).
- New gate: add it to `Gate` with its qiskit method name and its shift rule.
  A gate whose gradient follows neither the two-term nor the four-term rule
  also needs a new case in `Trainer._parameter_shift()`.

## `quantum.py`

The `QModel` class turns the events into circuits and runs them. For each
event, it builds one circuit: the amplitude embedding of the event, followed by
W(θ). These circuits are transpiled once, at the start; afterwards, each
evaluation only binds new θ values. `run()` returns P(qubit 0 = |1>) for each
circuit, in one of three modes:

- statevector: exact probability, computed without measurement;
- shots on the local Aer simulator: each circuit is measured `nm` times;
- shots on a real IBM Quantum computer, when `ibm_backend` is set.

`gate_sequence()` gives the gates of W(θ) with the trained values, which
`output.py` writes to `model.json`.

Where to change:

- Backend (simulator options, IBM connection): `make_backend()`.
- How a circuit is built or measured: `QModel.build()`.
- How P(signal) is read from the results: `QModel.run()` and
  `prob_qubit0_is_1()`.
- The number of qubits (`N_QUBITS = 4`) is fixed by the 12 features and the
  circuits: changing it means changing the data and `circuits.py` too.

## `training.py`

Training goes over all the events `n_iter` times, in mini-batches of
`batch_size` events. For each batch, the `Trainer` computes the loss, a
weighted binary cross-entropy (`WeightedBCE`), then its gradient with respect
to θ, and lets the Adam optimizer update θ.

The gradient comes from an exact formula, the parameter shift rule, which
reuses the same circuits with one value of θ shifted. A simple rotation needs 2
runs per value (shifts of ±π/2); a controlled rotation needs 4 (±π/2 and ±π).
The `Trainer` does not read the configuration: it receives the model, the loss
and the optimizer, so each one can be replaced without touching it.

Where to change:

- New loss: a class with `value(p, events)` and `derivative(p, events)`, like
  `WeightedBCE`, then given to the `Trainer` in `main.py`.
- Other optimizer: any object with `update_params([theta], [grad])`, such as
  sklearn's `SGDOptimizer`, given to the `Trainer` in `main.py`.
- When the losses are measured: `EvalSchedule`, filled from the `[model.evaluation]`
  table.

## `output.py`

At the end of the run, `output.py` writes three files to the run folder:

- `results.json`: the settings of the run, the final θ and the losses measured
  during training;
- `model.json`: the trained gates and 10 training events with their P(signal);
- a copy of `classifier.py`.

It then prints a summary: circuit, number of events, final θ, last losses and
run time.

Where to change:

- New value in `results.json`: `build_results()`.
- New value in `model.json`: `build_model()`. `classifier.py` reads this file,
  so a renamed or removed key must be changed there too.
- Name of the run folder: `run_dir_name()`.
- Final summary: `print_summary()`.

## `classifier.py`

A trained model is used through `classifier.py`, copied into each run folder
next to `model.json`. It does not import `pqc_hep` and only needs numpy, qiskit
and qiskit-aer, so the run folder can be sent on its own. Run from the run folder,
`python classifier.py events.csv` gives P(signal) for each event, and labels it
SIGNAL from 0.5. Without an argument, `python classifier.py` classifies the 10
events stored in `model.json` and checks that it finds the same P(signal) as at
the end of training.

Where to change: keep this file free of any `pqc_hep` import, since it is sent
without the rest of the code. A change to the encoding (`data.py`) or to
`model.json` (`output.py`) must be repeated here; `uv run pytest -m classifier`
checks that its predictions stay the same.
