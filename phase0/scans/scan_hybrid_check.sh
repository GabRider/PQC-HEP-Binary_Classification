#!/bin/bash
# Phase 0 - validation of the hybrid backend: reruns the 13 phase-0 runs with --backend hybrid.
# Same settings as the original runs (circuit 6, L = 1, seeds = 1): alpha scan, batch scan,
# raw (no z-score), full-batch verification (500 epochs) and the three 10k runs.
# Runs one after the other (sequential, so that the run times are comparable).
# Run folders go to phase0/results/ (prefix "hybrid_" or "raw_hybrid_", set automatically),
# logs to phase0/logs/ (prefix "hybrid_"). Takes about 15-20 min on the Mac.
# Compare with the original runs afterwards:  python phase0/analysis/compare_hybrid.py
# Usage:  bash phase0/scans/scan_hybrid_check.sh

cd "$(dirname "$0")/../results"
mkdir -p ../logs
CODE="../../Qiskit implementation"

run() {
    # $1 = log name, the other arguments are passed to PQC_phase0.py
    local LOG=$1; shift
    echo "=== start $LOG  ($(date +%H:%M)) ==="
    python "$CODE/PQC_phase0.py" --circuits-file "$CODE/circuits_library_Qiskit.py" \
        --circuit-number 6 --n-layers 1 --data-folder ../../data --backend hybrid "$@" \
        > ../logs/hybrid_$LOG.txt 2>&1
}

# 1,000 events, 50 epochs: alpha scan (b = 100)
run 1k_b100_a0.001          --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.001
run 1k_b100_a0.005          --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.005
run 1k_b100_a0.01           --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.01
run 1k_b100_a0.05           --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.05
run 1k_b100_a0.1            --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.1
# 1,000 events, 50 epochs: batch scan (alpha = 0.05)
run 1k_b10_a0.05            --max-events 1000  --n-iter 50  --batch-size 10   --alpha 0.05
run 1k_b1000_a0.05          --max-events 1000  --n-iter 50  --batch-size 1000 --alpha 0.05
# 1,000 events, 50 epochs, no z-score
run raw_1k_b100_a0.005      --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.005 --scaling none
run raw_1k_b100_a0.05       --max-events 1000  --n-iter 50  --batch-size 100  --alpha 0.05  --scaling none
# full-batch verification: 1,000 events, b = 1000, 500 epochs
run verif_1k_b1000_500ep    --max-events 1000  --n-iter 500 --batch-size 1000 --alpha 0.05
# 10,000 events, 50 epochs (alpha = 0.05)
run 10k_b2000               --max-events 10000 --n-iter 50  --batch-size 2000 --alpha 0.05
run 10k_b1000               --max-events 10000 --n-iter 50  --batch-size 1000 --alpha 0.05
run 10k_b100                --max-events 10000 --n-iter 50  --batch-size 100  --alpha 0.05

echo "=== scan finished ($(date +%H:%M)) ==="
