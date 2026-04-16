#!/bin/bash
# Launch all 3 experiments in parallel on Brev GPU
export PATH=/home/shadeform/.local/bin:$PATH

echo "[launcher] Starting 3 experiments in parallel..."

nohup python3 /tmp/exp_a_values.py   > /tmp/out_a.log 2>&1 &
PID_A=$!
echo "[launcher] Exp A (values)   PID=$PID_A"

nohup python3 /tmp/exp_b_hadamard.py > /tmp/out_b.log 2>&1 &
PID_B=$!
echo "[launcher] Exp B (hadamard) PID=$PID_B"

nohup python3 /tmp/exp_c_fused.py    > /tmp/out_c.log 2>&1 &
PID_C=$!
echo "[launcher] Exp C (fused)    PID=$PID_C"

echo "[launcher] Waiting for all 3..."
wait $PID_A && echo "[launcher] Exp A DONE" || echo "[launcher] Exp A FAILED"
wait $PID_B && echo "[launcher] Exp B DONE" || echo "[launcher] Exp B FAILED"
wait $PID_C && echo "[launcher] Exp C DONE" || echo "[launcher] Exp C FAILED"
echo "[launcher] All done."
