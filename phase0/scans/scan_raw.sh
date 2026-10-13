#!/bin/bash
# Phase 0 - same runs WITHOUT standardisation (--scaling none), to compare with z-score.
# Circuit 6, L = 1, 1000 training events, 50 epochs, b = 100, seeds = 1.
# Output folders are prefixed with "raw_".
# Usage (from anywhere):  caffeinate -i bash scan_raw.sh

N_EVENTS=1000
N_ITER=50
BATCH=100
ALPHAS="0.005 0.05"

# Run folders are written to phase0/results/, logs to phase0/logs/.
cd "$(dirname "$0")/../results"
mkdir -p ../logs
CODE="../../Qiskit implementation"

for ALPHA in $ALPHAS; do
    echo "=== start raw, alpha = $ALPHA  ($(date +%H:%M)) ==="
    python "$CODE/PQC_phase0.py" --circuits-file "$CODE/circuits_library_Qiskit.py" \
        --circuit-number 6 --n-layers 1 --data-folder ../../data \
        --max-events $N_EVENTS --n-iter $N_ITER --batch-size $BATCH --alpha $ALPHA \
        --scaling none \
        > ../logs/raw_alpha_$ALPHA.txt 2>&1 &
done
wait

echo "=== scan finished ($(date +%H:%M)) ==="
