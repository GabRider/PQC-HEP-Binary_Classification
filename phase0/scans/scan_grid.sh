#!/bin/bash
# Grid alpha x b (replaces the phase-0 scans in the report).
# Circuit 6, L = 1, hybrid backend, z-score, 100,000 training and 100,000 validation events,
# both balanced (50,000 S / 50,000 B), 50 epochs for every b, best epoch by validation loss.
# 6 alphas x 4 batch sizes x 3 theta seeds = 72 runs, one after the other.
# Run folders go to phase0/results/grid/, logs to phase0/logs/grid/.
# If the grid stops halfway, restart it with only the missing values (see below).
# Usage:             bash phase0/scans/scan_grid.sh
# Only some values:  ALPHAS="0.01 0.03" SEEDS="2" bash phase0/scans/scan_grid.sh

N_EVENTS=${N_EVENTS:-100000}
N_ITER=${N_ITER:-50}
ALPHAS=${ALPHAS:-"0.001 0.003 0.01 0.03 0.1 0.3"}
BATCHES=${BATCHES:-"1000 10000 20000 50000"}
SEEDS=${SEEDS:-"1 2 3"}

cd "$(dirname "$0")/../results"
mkdir -p grid ../logs/grid
cd grid
CODE="../../../Qiskit implementation"

run() {
    # $1 = alpha, $2 = batch size, $3 = theta seed
    local A=$1 B=$2 S=$3
    local NAME="b${B}_a${A}_s${S}"
    echo "=== start $NAME  ($(date +%H:%M:%S)) ==="
    python "$CODE/PQC_phase0.py" --circuits-file "$CODE/circuits_library_Qiskit.py" \
        --circuit-number 6 --n-layers 1 --data-folder ../../../data --backend hybrid \
        --max-events $N_EVENTS --n-iter $N_ITER --batch-size $B --alpha $A --theta-seed $S \
        --run-tag grid > ../../logs/grid/$NAME.txt 2>&1 \
        || echo "=== FAILED $NAME, see logs/grid/$NAME.txt ==="
}

echo "=== grid start ($(date +%H:%M:%S)) ==="
for S in $SEEDS; do
    for A in $ALPHAS; do
        for B in $BATCHES; do
            run $A $B $S
        done
    done
done
echo "=== grid finished ($(date +%H:%M:%S)) ==="