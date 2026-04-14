#!/bin/bash
# CQL-baseline testing: 5 seeds x 5 scenarios x 100 ep after training is done
# Run in tmux: tmux new-session -d -s cql_bl_test "bash scripts/launch_cql_baseline_test.sh"
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.

# Wait for all 5 seeds to complete training
echo "[$(date '+%H:%M:%S')] Waiting for CQL-baseline models..."
for seed in 0; do
  while [ ! -f "data/channel_generality/cql_baseline/models/expert_seed${seed}/model.pt" ]; do
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] seed ${seed} ready"
done

echo "[$(date '+%H:%M:%S')] All 5 seeds done. Starting testing..."

for seed in 0 1 2 3 4; do
  MODEL="data/channel_generality/cql_baseline/models/expert_seed0/model.pt"
  for scen in 5 6 7 8 9; do
    echo "[$(date '+%H:%M:%S')] Testing seed=$seed scenario=$scen"
    # 145 = PPO-baseline / dt_baseline_v2 flat obs; default 45 is for cql_v2 wrapper only
    pixi run python -u src/basic_apis/cql_baseline/test_cql.py \
      --model "$MODEL" \
      --scenario $scen --seed $seed \
      --n_episodes 100 \
      --variant expert \
      --save_root data/channel_generality/cql_baseline \
      --obs_dim 145 \
      --init_episode 0 --max_episode 100
  done
  echo "[$(date '+%H:%M:%S')] seed $seed ALL scenarios DONE"
done

echo "[$(date '+%H:%M:%S')] CQL-baseline test ALL DONE"
