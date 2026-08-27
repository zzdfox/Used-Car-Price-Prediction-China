#!/bin/bash
# Round 5: expensive-car feature A/B (exp13, CPU) -> heterogeneous NN DCN/FTT (exp14, GPU) -> 12-member blend
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
LPY=~/limix-venv/bin/python
TS="taskset -c 5-9,15-19"
LOG=user_data/run_round5.log
rm -f user_data/run_round5.done
{
  echo "===== exp13 expensive-car feature A/B: $(date) ====="
  $TS $PY src/exp13_feat.py
  echo "===== exp14 search (dcn/ftt): $(date) ====="
  $TS $LPY src/exp14_nn.py search
  echo "===== blend eval: $(date) ====="
  $TS $PY src/blend_report.py eval
  echo "===== round5 done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_round5.done
