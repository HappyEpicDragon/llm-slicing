#!/bin/bash
# Multi-seed testing launcher for all methods on s5-s9
# Usage: Run individual sections as tmux commands, or source this file.
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.

# ============================================================
# B1: IDT-v2 E4 — 5 seeds x 100 ep x s5-s9
# Expected time: ~4h per GPU (5 seeds sequential)
# ============================================================
launch_e4_5seed() {
  local GPU=$1
  echo "=== E4 5-seed on cuda:${GPU} ==="
  pixi run sim test_dt_v2 \
    test_dt_v2.model_path=data/channel_generality/dt_v2_8exp/dt_model_E4_slice_attn/epoch_60.pth \
    test_dt_v2.meta_path=data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json \
    test_dt_v2.encoder_type=slice_attn \
    test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
    test_dt_v2.test_scenarios=[5,6,7,8,9] \
    test_dt_v2.test_seeds=[0,1,2,3,4] \
    test_dt_v2.n_episodes=100 \
    test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
    test_dt_v2.save_root=data/channel_generality/dt_v2_8exp/eval_ood_5seed/E4 \
    test_dt_v2.device=cuda:${GPU}
  echo "E4 5-seed DONE"
}

# ============================================================
# B2: PPO-teacher (joint SliceAttn) — 5 seeds x 100 ep x s5-s9
# Expected time: ~4h per GPU
# ============================================================
launch_ppo_teacher() {
  local GPU=$1
  echo "=== PPO-teacher on cuda:${GPU} ==="
  pixi run sim test_ppo_v2 \
    test_ppo_v2.model_path=data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn/best_model/best_model.zip \
    test_ppo_v2.test_scenarios=[5,6,7,8,9] \
    test_ppo_v2.test_seeds=[0,1,2,3,4] \
    test_ppo_v2.n_episodes=100 \
    test_ppo_v2.init_episode=0 test_ppo_v2.max_episode=100 \
    test_ppo_v2.save_root=data/channel_generality/ppo_teacher_5seed \
    test_ppo_v2.save_metric_key=eval_ood \
    test_ppo_v2.device=cuda:${GPU}
  echo "PPO-teacher DONE"
}

# ============================================================
# A5: CQL testing — 5 seeds x 100 ep x s5-s9
# Requires CQL training to be finished first
# ============================================================
launch_cql_test() {
  local GPU=$1
  echo "=== CQL test on cuda:${GPU} ==="
  # CQL test doesn't take a device arg directly; it runs on CPU for inference
  # But env simulation is CPU-bound anyway
  for seed in 0 1 2 3 4; do
    MODEL="data/channel_generality/cql_v2/models/expert_seed${seed}/model.pt"
    if [ ! -f "$MODEL" ]; then
      echo "SKIP seed $seed: $MODEL not found"
      continue
    fi
    for scen in 5 6 7 8 9; do
      pixi run python -u src/basic_apis/cql_baseline/test_cql.py \
        --model "$MODEL" \
        --scenario $scen --seed $seed \
        --n_episodes 100 --variant expert \
        --save_root data/channel_generality/cql_v2 \
        --init_episode 0 --max_episode 100
    done
    echo "CQL seed $seed DONE"
  done
  echo "CQL test ALL DONE"
}

echo "Available functions:"
echo "  launch_e4_5seed <GPU_ID>"
echo "  launch_ppo_teacher <GPU_ID>"
echo "  launch_cql_test <GPU_ID>"
