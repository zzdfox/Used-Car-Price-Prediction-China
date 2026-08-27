#!/bin/bash
# Round 3: NN deep dive (name hash / wider net / 3-seed average) -> re-blend -> regenerate testB
# Chained: waits for run_round2_final to finish
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
LPY=~/limix-venv/bin/python
TS="taskset -c 5-9,15-19"
LOG=user_data/run_round3.log
rm -f user_data/run_round3.done
while [ ! -f user_data/run_round2_final.done ]; do sleep 60; done
{
  cp user_data/pred_eval_nn.npz user_data/bak_nn_eval_seed42.npz
  cp user_data/pred_final_nn.npz user_data/bak_nn_final_seed42.npz
  echo "===== exp11 search (eval): $(date) ====="
  $TS $LPY src/exp11_nn.py search
  echo "===== blend eval (nn=3seed): $(date) ====="
  $TS $PY src/blend_report.py eval
  echo "===== exp11 final: $(date) ====="
  $TS $LPY src/exp11_nn.py final
  echo "===== blend final: $(date) ====="
  $TS $PY src/blend_report.py final
  cp prediction_result/predictions.csv prediction_result/predictions_testB.csv
  echo "===== round3 done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_round3.done
