#!/bin/bash
# Round 6, stacking more capacity: DCN 5-seed + cat tuning (cat_b) -> re-blend -> regenerate testB
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
LPY=~/limix-venv/bin/python
TS="taskset -c 5-9,15-19"
LOG=user_data/run_round6.log
rm -f user_data/run_round6.done
{
  echo "===== exp14 dcn 5-seed (eval): $(date) ====="
  $TS $LPY src/exp14_nn.py search dcn
  echo "===== exp16 cat tuning search: $(date) ====="
  $TS $PY src/exp16_cat.py search
  echo "===== blend eval: $(date) ====="
  $TS $PY src/blend_report.py eval
  echo "===== exp14 dcn 5-seed (final): $(date) ====="
  $TS $LPY src/exp14_nn.py final dcn
  echo "===== exp16 cat_b final: $(date) ====="
  $TS $PY src/exp16_cat.py final
  echo "===== blend final: $(date) ====="
  $TS $PY src/blend_report.py final
  cp prediction_result/predictions.csv prediction_result/predictions_testB.csv
  echo "===== round6 done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_round6.done
