#!/bin/bash
# Round 4, pushing toward 400: NN variant search + 5-seed average + nn2 free member + lgb_d -> re-blend, regenerate testB
set -e
cd ~/used-car-prediction
PY=~/ucp-venv/bin/python
LPY=~/limix-venv/bin/python
TS="taskset -c 5-9,15-19"
LOG=user_data/run_round4.log
rm -f user_data/run_round4.done
{
  echo "===== exp12 search+5seed (eval): $(date) ====="
  $TS $LPY src/exp12_nn.py search
  # nn2: exp10 base-architecture single-seed model, restored from backup as a standalone blend member (zero cost)
  cp user_data/bak_nn_eval_seed42.npz user_data/pred_eval_nn2.npz
  cp user_data/bak_nn_final_seed42.npz user_data/pred_final_nn2.npz
  echo "===== lgb_d eval: $(date) ====="
  $TS $PY src/main.py --stage eval --models lgb_d --folds 5
  echo "===== blend eval (10 members): $(date) ====="
  $TS $PY src/blend_report.py eval
  echo "===== exp12 final 5seed: $(date) ====="
  $TS $LPY src/exp12_nn.py final
  echo "===== lgb_d final: $(date) ====="
  $TS $PY src/main.py --stage final --test B --models lgb_d --folds 5
  echo "===== blend final (10 members): $(date) ====="
  $TS $PY src/blend_report.py final
  cp prediction_result/predictions.csv prediction_result/predictions_testB.csv
  echo "===== round4 done: $(date) ====="
} > "$LOG" 2>&1
touch user_data/run_round4.done
