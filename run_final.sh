#!/bin/bash
# Chained job: wait for eval baseline + exp6 (run_all.sh) to finish,
# then retrain on the full 150k rows and produce the testB submission, finally run exp7 feature pruning.
# Usage: nohup bash run_final.sh </dev/null >/dev/null 2>&1 &
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
TS="taskset -c 5-9,15-19"      # GB10 performance cores
LOG=user_data/run_final_20260826.log
rm -f user_data/run_final_20260826.done
while [ ! -f user_data/run_20260826.done ]; do sleep 60; done
{
  echo "===== final (--stage final --test B) start: $(date) ====="
  $TS $PY src/main.py --stage final --test B --folds 5
  echo "===== exp7 (feature pruning) start: $(date) ====="
  $TS $PY src/exp7.py
  echo "===== final chain done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_final_20260826.done
