#!/bin/bash
# Serial run: 5-model eval baseline (fixed cat/xgb losses + 216-dim features) -> exp6 A/B
# Usage: nohup bash run_all.sh >/dev/null 2>&1 &
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
TS="taskset -c 5-9,15-19"      # GB10 performance cores, see doc/FINDINGS.md section 1
LOG=user_data/run_20260826.log
rm -f user_data/run_20260826.done
{
  echo "===== eval baseline start: $(date) ====="
  $TS $PY src/main.py --stage eval --folds 5
  echo "===== exp6 (reg_month_missing A/B) start: $(date) ====="
  $TS $PY src/exp6.py
  echo "===== all done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_20260826.done
