#!/bin/bash
# Phase 0 - batch-size scan.
# Circuit 6, L = 1, 1000 training events, 50 epochs, alpha = 0.05, z-score, seeds = 1.
# (b = 100 is the alpha = 0.05 run of scan_alpha.sh.)
# Usage (from anywhere):  caffeinate -i bash scan_batch.sh

# --- Scan settings (quick test: N_EVENTS=200, N_ITER=1, BATCHES="10 200") ---
N_EVENTS=1000
N_ITER=50
ALPHA=0.05
BATCHES="10 1000"

# Run folders are written to phase0/results/, logs to phase0/logs/.
cd "$(dirname "$0")/../results"
mkdir -p ../logs
CODE="../../Qiskit implementation"

for B in $BATCHES; do
    echo "=== start batch = $B  ($(date +%H:%M)) ==="
    python "$CODE/PQC_phase0.py" --circuits-file "$CODE/circuits_library_Qiskit.py" \
        --circuit-number 6 --n-layers 1 --data-folder ../../data \
        --max-events $N_EVENTS --n-iter $N_ITER --batch-size $B --alpha $ALPHA \
        > ../logs/batch_$B.txt 2>&1 &
done
wait

echo "=== scan finished ($(date +%H:%M)) ==="
