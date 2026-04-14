#!/bin/bash
# Automated scheduler: watches for GPU availability and launches test tasks
# Usage: tmux new-session -d -s scheduler "bash scripts/auto_scheduler.sh"
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_file() {
  local path=$1
  while [ ! -f "$path" ]; do sleep 30; done
}

# ============================================================
# Phase 1: Wait for CQL seed 3 (GPU3) to finish → launch CQL test (CPU)
# ============================================================
log "Phase 1: Waiting for CQL seed 4 to complete (GPU3)..."
while ! ls data/channel_generality/cql_v2/models/expert_seed4/model.pt 2>/dev/null; do
  sleep 60
done
log "CQL seed 4 done. Launching CQL testing..."

tmux new-session -d -s cql_test "bash -c '
cd /root/decision_transformer_slicing && export PYTHONPATH=.
echo \"=== CQL Testing: 5 seeds x 5 scenarios x 100 ep ===\"
for seed in 0 1 2 3 4; do
  MODEL=\"data/channel_generality/cql_v2/models/expert_seed\${seed}/model.pt\"
  if [ ! -f \"\$MODEL\" ]; then
    echo \"SKIP seed \$seed: model not found\"
    continue
  fi
  for scen in 5 6 7 8 9; do
    echo \"Testing seed=\$seed scenario=\$scen...\"
    pixi run python -u src/basic_apis/cql_baseline/test_cql.py \
      --model \"\$MODEL\" --scenario \$scen --seed \$seed \
      --n_episodes 100 --variant expert \
      --save_root data/channel_generality/cql_v2 \
      --init_episode 0 --max_episode 100
  done
  echo \"CQL seed \$seed testing DONE\"
done
echo \"CQL test ALL DONE\"
exec bash'"
log "CQL test session launched."

# ============================================================
# Phase 2: Wait for E5r (GPU0) to finish → launch E4 5-seed (GPU0)
# ============================================================
log "Phase 2: Waiting for E5r training to complete (GPU0)..."
while tmux has-session -t dt_E5r 2>/dev/null && \
      ! tmux capture-pane -t dt_E5r -p 2>/dev/null | grep -q "GPU 0 ALL DONE"; do
  sleep 60
done
log "E5r done. Launching E4 5-seed test on GPU0..."

tmux new-session -d -s e4_5seed "bash -c '
cd /root/decision_transformer_slicing && export PYTHONPATH=.
echo \"=== E4 5-seed on cuda:0 ===\"
pixi run sim test_dt_v2 \
  test_dt_v2.model_path=data/channel_generality/dt_v2_8exp/dt_model_E4_slice_attn/epoch_60.pth \
  test_dt_v2.meta_path=data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json \
  test_dt_v2.encoder_type=slice_attn \
  test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
  test_dt_v2.test_scenarios=[5,6,7,8,9] \
  \"test_dt_v2.test_seeds=[0,1,2,3,4]\" \
  test_dt_v2.n_episodes=100 \
  test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
  test_dt_v2.save_root=data/channel_generality/dt_v2_8exp/eval_ood_5seed/E4 \
  test_dt_v2.device=cuda:0
echo \"E4 5-seed DONE\"
exec bash'"
log "E4 5-seed session launched on GPU0."

# ============================================================
# Phase 3: Wait for E6r (GPU1) to finish → launch PPO-teacher (GPU1)
# ============================================================
log "Phase 3: Waiting for E6r training to complete (GPU1)..."
while tmux has-session -t dt_E6r 2>/dev/null && \
      ! tmux capture-pane -t dt_E6r -p 2>/dev/null | grep -q "GPU 1 ALL DONE"; do
  sleep 60
done
log "E6r done. Launching PPO-teacher test on GPU1..."

tmux new-session -d -s ppo_teacher "bash -c '
cd /root/decision_transformer_slicing && export PYTHONPATH=.
echo \"=== PPO-teacher 5-seed on cuda:1 ===\"
pixi run sim test_ppo_v2 \
  test_ppo_v2.model_path=data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn/best_model/best_model.zip \
  test_ppo_v2.test_scenarios=[5,6,7,8,9] \
  \"test_ppo_v2.test_seeds=[0,1,2,3,4]\" \
  test_ppo_v2.n_episodes=100 \
  test_ppo_v2.init_episode=0 test_ppo_v2.max_episode=100 \
  test_ppo_v2.save_root=data/channel_generality/ppo_teacher_5seed \
  test_ppo_v2.save_metric_key=eval_ood \
  test_ppo_v2.device=cuda:1
echo \"PPO-teacher DONE\"
exec bash'"
log "PPO-teacher session launched on GPU1."

log "All Phase 1-3 tasks dispatched. Monitor with: tmux ls"
