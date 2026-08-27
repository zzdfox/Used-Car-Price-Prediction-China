#!/bin/bash
# Round 2 optimization: seed-average members (lgb_c, xgb_b) + embedding-NN (exp10) -> 8-member blend
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
LPY=~/limix-venv/bin/python
TS="taskset -c 5-9,15-19"
LOG=user_data/run_round2.log
rm -f user_data/run_round2.done
{
  echo "===== lgb_c eval: $(date) ====="
  $TS $PY src/main.py --stage eval --models lgb_c --folds 5
  echo "===== xgb_b eval: $(date) ====="
  $TS $PY src/main.py --stage eval --models xgb_b --folds 5
  echo "===== nn (exp10) eval: $(date) ====="
  $TS $LPY src/exp10_nn.py eval
  echo "===== blend (8 members): $(date) ====="
  $TS $PY src/blend_report.py eval
  echo "===== round2 done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_round2.done
