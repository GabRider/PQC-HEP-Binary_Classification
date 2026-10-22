#!/bin/bash
# Phase 0 - runs with 10,000 training events: batch size 1 % (b = 100), 10 % (b = 1000), 20 % (b = 2000).
# Circuit 6, L = 1, 50 epochs, alpha = 0.05, z-score, seeds = 1. Same computing cost for all runs.
# Runs in parallel; run folders go to phase0/results/ (prefix "10k_"), logs to phase0/logs/.
# Usage:  bash phase0/scans/scan_10k.sh
# Quick test:  N_EVENTS=200 N_ITER=1 TAG=TEST BATCHES="100" bash phase0/scans/scan_10k.sh

N_EVENTS=${N_EVENTS:-10000}
N_ITER=${N_ITER:-50}
ALPHA=${ALPHA:-0.05}
BATCHES=${BATCHES:-"100 1000 2000"}
TAG=${TAG:-10k}

cd "$(dirname "$0")/../results"
mkdir -p ../logs
CODE="../../Qiskit implementation"

for B in $BATCHES; do
    echo "=== start batch = $B  ($(date +%H:%M)) ==="
    python "$CODE/PQC_phase0.py" --circuits-file "$CODE/circuits_library_Qiskit.py" \
        --circuit-number 6 --n-layers 1 --data-folder ../../data \
        --max-events $N_EVENTS --n-iter $N_ITER --batch-size $B --alpha $ALPHA \
        --run-tag $TAG \
        > ../logs/${TAG}_batch_$B.txt 2>&1 &
done
wait

echo "=== scan finished ($(date +%H:%M)) ==="
