"""
Author: Adriano Pinto Claro da Fonseca
Email: apcf@topfonseca.com
Institutional email: adriano.fonseca@tecnico.ulisboa.pt
Affiliation: Student at Instituto Superior Técnico, Universidade de Lisboa

See Readme.md and adriano_code/README.md for more information.
Sphinx documentation will be implemented after refactoring.
"""

import argparse
import datetime
import sys
import time

import numpy as np
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
from tqdm import tqdm

from pqc_hep import circuits, output
from pqc_hep.circuits import ShiftRule
from pqc_hep.config import load as load_config
from pqc_hep.data import (
    load_events,
    normalise_weights,
    order_events,
    select_events,
    to_amplitudes,
)
from pqc_hep.quantum import QModel, make_backend
from pqc_hep.training import (
    EvalSchedule,
    Trainer,
    TrainingSet,
    WeightedBCE,
    measure_loss,
    resolve_batch_size,
)


def _show_loss(loss):
    return "--------" if loss is None else f"{loss:.6f}"


def record_eval(evals, key, batch_loss, global_loss, n_iter):
    """Save the losses of one evaluation ("start" or the pass number) in
    evals["batch"] and evals["global"], and print them on one line;
    None = not measured at this evaluation."""
    if batch_loss is not None:
        evals["batch"][key] = float(batch_loss)
    if global_loss is not None:
        evals["global"][key] = float(global_loss)

    width = max(5, len(str(n_iter)))  # "start" is 5 chars: aligns [n=start], [n=  N]
    tqdm.write(
        f"  [n={key:>{width}}]  "
        f"batch = {_show_loss(batch_loss)}   GLOBAL = {_show_loss(global_loss)}"
    )


def parse_args():
    """Command-line options of `python main.py`."""
    parser = argparse.ArgumentParser(
        description="PQC mini-batch training pipeline for HEP binary "
        "classification (Qiskit)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="TOML configuration file; built-in defaults are used when omitted.",
    )
    return parser.parse_args()


def main(config_path=None):
    """Train with the TOML configuration at config_path (built-in defaults when
    None) and write the outputs to a timestamped folder."""
    t_start = time.time()

    # =========================================================================
    # INPUT HANDLING
    # =========================================================================
    # Reads the TOML run configuration, picks the circuit, prepares the
    # evaluation schedule, and prints the effective configuration.

    cfg = load_config(config_path)

    circuit_number = cfg.model.circuit.circuit_number
    n_layers = cfg.model.circuit.n_layers
    n_iter = cfg.model.training.n_iter

    circuit = circuits.CIRCUITS[circuit_number]
    n_params = len(circuit.shift_rules(n_layers))  # one theta value per rotation

    eval_schedule = EvalSchedule.from_config(cfg.model.evaluation, n_iter)
    global_at = sorted(eval_schedule.global_passes)
    batch_at = sorted(eval_schedule.batch_passes)

    rule = "=" * 62
    print(f"""
{rule}
  EFFECTIVE CONFIGURATION
{rule}
{cfg}
[extra]
  circuit                   : [{circuit_number}] {circuit.description}
  parameters                : {n_params}
  global_eval_at            : start={eval_schedule.global_start}, n_iter={global_at}
  batch_eval_at             : start={eval_schedule.batch_start}, n_iter={batch_at}
{rule}
""")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    # Reads the signal and background CSV files, selects and orders the events,
    # normalises their weights, and resolves batch_size to an event count.

    print("Loading data...")
    sig, bkg = load_events(cfg.data.sig_path, cfg.data.bkg_path)
    rng = np.random.default_rng(seed=cfg.seeds.data_shuffle)
    events = select_events(
        sig, bkg, cfg.data.event_limit, cfg.data.balance_classes, rng
    )
    events = order_events(events, cfg.data.event_order)
    events = normalise_weights(events)
    n_events = len(events)
    n_loaded = len(sig) + len(bkg)
    n_sig = int((events.labels == 1).sum())
    n_bkg = int((events.labels == 0).sum())

    print(f"  Loaded : {n_loaded} events ({len(sig)} signal, {len(bkg)} background)")
    print(f"  Using  : {n_events} events ({n_sig} signal, {n_bkg} background)")
    print(f"  Weight normalisation: sum(w_S) = sum(w_B) = {n_bkg}")

    batch_size = resolve_batch_size(cfg.model.training.batch_size, n_events)
    if isinstance(cfg.model.training.batch_size, float):
        print(
            f"  Batch size (resolved) : {batch_size}  "
            f"(= {batch_size / n_events * 100:.1f}% of {n_events} events)"
        )

    n_batches_per_iter = n_events // batch_size
    print(f"  Batches per N_iter : {n_batches_per_iter}  (batch_size={batch_size})\n")

    # =========================================================================
    # QUANTUM PIPELINE
    # =========================================================================
    # The circuits and their execution are in quantum.py (QModel); the loss and
    # the gradient via the parameter shift rule are in training.py.

    # Local AerSimulator, or a real IBM Quantum computer when ibm_backend is set.
    backend, sampler = make_backend(cfg.model.backend, cfg.seeds.simulator)
    if cfg.model.backend.ibm_backend is not None:
        print(f"  Using real IBM Quantum hardware: {cfg.model.backend.ibm_backend}")
    shots = None if cfg.model.backend.use_statevector else cfg.model.backend.nm
    qmodel = QModel(
        circuit, n_layers, backend, shots, sampler, preprocess=to_amplitudes
    )

    # =========================================================================
    # TRAINING SETUP
    # =========================================================================
    # Build and transpile one full circuit per event (embedding + variational
    # form). All N_iter passes reuse these circuits by binding the theta values.

    print(f"Building and transpiling circuits ({n_events} events)...")
    all_circuits = qmodel.build(events.features)
    print(f"  Done: {len(all_circuits)} circuits transpiled.\n")

    n_two = qmodel.shift_rules.count(ShiftRule.TWO_TERM)
    n_four = qmodel.shift_rules.count(ShiftRule.FOUR_TERM)
    print(
        f"  Shift rule classification: {n_two} two-term (Rx/Ry/Rz), "
        f"{n_four} four-term (CRx/CRy/CRz)\n"
    )

    # Evaluation losses, filled by record_eval().
    # Key: "start" or str(pass number); value: loss (float).
    evals = {"batch": {}, "global": {}}

    adam = cfg.model.adam
    sim_info = (
        "statevector (exact)"
        if cfg.model.backend.use_statevector
        else f"Nm={cfg.model.backend.nm} shots"
    )
    print(
        f"Starting training: {n_iter} passes x {n_batches_per_iter} batches "
        f"x {batch_size} events/batch"
    )
    print(f"  Simulation : {sim_info}")
    print(
        f"  Adam       : sklearn AdamOptimizer, Kingma & Ba 2014  "
        f"alpha={adam.alpha}  beta1={adam.beta1}  beta2={adam.beta2}\n"
    )

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    # The config chooses the components; the Trainer only uses them.

    # Initial theta uniformly in [-pi, pi].
    theta_rng = np.random.default_rng(seed=cfg.seeds.theta_init)
    theta0 = theta_rng.uniform(-np.pi, np.pi, n_params)
    optimizer = AdamOptimizer(
        params=[theta0],
        learning_rate_init=adam.alpha,
        beta_1=adam.beta1,
        beta_2=adam.beta2,
        epsilon=adam.epsilon,
    )
    loss = WeightedBCE()
    trainer = Trainer(qmodel, loss, optimizer, theta0, batch_size)
    data = TrainingSet(events, all_circuits)

    # "start" evaluations: initial random theta, before any optimizer step.
    if eval_schedule.batch_start or eval_schedule.global_start:
        record_eval(
            evals,
            "start",
            measure_loss(qmodel, data[:batch_size], trainer.theta, loss, batch_size)
            if eval_schedule.batch_start
            else None,
            measure_loss(qmodel, data, trainer.theta, loss, batch_size)
            if eval_schedule.global_start
            else None,
            n_iter,
        )
        print()

    with tqdm(total=n_iter * n_batches_per_iter, unit="batch") as bar:
        for n in range(1, n_iter + 1):
            bar.set_description(f"Pass {n:{len(str(n_iter))}d}/{n_iter}")
            reshuffle = cfg.data.reshuffle_between_passes and n > 1
            seed = cfg.seeds.pass_reshuffle + n if reshuffle else None
            for batch in trainer.batches(data, seed):
                last_batch_loss = trainer.step(batch)
                bar.update()
            # Batch evaluation: loss of the pass's last batch, before its step.
            # Global evaluation: loss on all the events.
            if n in eval_schedule.batch_passes or n in eval_schedule.global_passes:
                record_eval(
                    evals,
                    str(n),
                    last_batch_loss if n in eval_schedule.batch_passes else None,
                    measure_loss(qmodel, data, trainer.theta, loss, batch_size)
                    if n in eval_schedule.global_passes
                    else None,
                    n_iter,
                )
    theta_final = trainer.theta.tolist()
    gate_sequence = qmodel.gate_sequence(theta_final)

    # =========================================================================
    # OUTPUT GENERATION
    # =========================================================================
    # Writes results.json, model.json and a copy of classifier.py to a
    # timestamped folder, then prints the final summary (output.py).

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = output.run_dir_name(cfg, batch_size, timestamp)
    results = output.build_results(
        cfg,
        timestamp,
        len(sig),
        len(bkg),
        n_events,
        batch_size,
        eval_schedule,
        theta_final,
        gate_sequence,
        evals["global"],
        evals["batch"],
    )
    # model.json examples: the first 5 signal and 5 background training events.
    example_idx = np.array(
        [
            *np.flatnonzero(events.labels == 1)[:5],
            *np.flatnonzero(events.labels == 0)[:5],
        ]
    )
    example_p = qmodel.run([all_circuits[i] for i in example_idx], theta_final)
    model = output.build_model(
        cfg, timestamp, gate_sequence, events[example_idx], example_p
    )
    output.save(output_dir, results, model)
    output.print_summary(results, output_dir, time.time() - t_start)


if __name__ == "__main__":
    # Errors caused by the user's config or files: a message without traceback.
    try:
        main(parse_args().config)
    except (OSError, ValueError, ImportError) as e:
        sys.exit(f"Error: {e}")
