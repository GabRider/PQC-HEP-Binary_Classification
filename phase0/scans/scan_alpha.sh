#!/bin/bash
# Phase 0 - learning-rate scan.
# Circuit 6, L = 1, 1000 training events, 50 epochs, b = 100, z-score, seeds = 1.
# Runs are launched in parallel (one per value of ALPHAS); logs go to phase0/logs/.
# Usage (from anywhere):  caffeinate -i bash scan_alpha.sh

# --- Scan settings (quick test: N_EVENTS=200, N_ITER=1, ALPHAS="0.005 0.05") ---
N_EVENTS=1000
N_ITER=50
BATCH=100
ALPHAS="0.001 0.005 0.01 0.05 0.1"

# Run folders are written to phase0/results/, logs to phase0/logs/.
cd "$(dirname "$0")/../results"
mkdir -p ../logs
CODE="../../Qiskit implementation"

for ALPHA in $ALPHAS; do
    echo "=== start alpha = $ALPHA  ($(date +%H:%M)) ==="
    python "$CODE/PQC_phase0.py" --circuits-file "$CODE/circuits_library_Qiskit.py" \
        --circuit-number 6 --n-layers 1 --data-folder ../../data \
        --max-events $N_EVENTS --n-iter $N_ITER --batch-size $BATCH --alpha $ALPHA \
        > ../logs/alpha_$ALPHA.txt 2>&1 &
done
wait

echo "=== scan finished ($(date +%H:%M)) ==="
