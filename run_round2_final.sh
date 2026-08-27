#!/bin/bash
# Round 2 final: retrain new members on full data (lgb_c, xgb_b, nn) -> 8-member blend, regenerate testB submission
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
LPY=~/limix-venv/bin/python
TS="taskset -c 5-9,15-19"
LOG=user_data/run_round2_final.log
rm -f user_data/run_round2_final.done
{
  echo "===== lgb_c final: $(date) ====="
  $TS $PY src/main.py --stage final --test B --models lgb_c --folds 5
  echo "===== xgb_b final: $(date) ====="
  $TS $PY src/main.py --stage final --test B --models xgb_b --folds 5
  echo "===== nn final: $(date) ====="
  $TS $LPY src/exp10_nn.py final
  echo "===== blend final (8 members): $(date) ====="
  $TS $PY src/blend_report.py final
  cp prediction_result/predictions.csv prediction_result/predictions_testB.csv
  echo "===== round2 final done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_round2_final.done
