#!/bin/bash
# Parallel 5-seed testing: split by scenario across 4 GPUs
# E4 DT (5 scenarios) + PPO-teacher (5 scenarios) = 10 processes across 4 GPUs
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.

# GPU assignment: 2-3 processes per GPU
# GPU0: E4-s5 + PPO-s5
# GPU1: E4-s6 + PPO-s6
# GPU2: E4-s7 + E4-s8 + PPO-s7
# GPU3: E4-s9 + PPO-s8 + PPO-s9

run_e4_scenario() {
  local SCEN=$1
  local GPU=$2
  echo "[$(date '+%H:%M:%S')] E4 scenario $SCEN on cuda:$GPU START"
  pixi run sim test_dt_v2 \
    test_dt_v2.model_path=data/channel_generality/dt_v2_8exp/dt_model_E4_slice_attn/epoch_60.pth \
    test_dt_v2.meta_path=data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json \
    test_dt_v2.encoder_type=slice_attn \
    test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
    test_dt_v2.test_scenarios=[$SCEN] \
    "test_dt_v2.test_seeds=[0,1,2,3,4]" \
    test_dt_v2.n_episodes=100 \
    test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
    test_dt_v2.save_root=data/channel_generality/dt_v2_8exp/eval_ood_5seed/E4 \
    test_dt_v2.device=cuda:$GPU
  echo "[$(date '+%H:%M:%S')] E4 scenario $SCEN DONE"
}

run_ppo_scenario() {
  local SCEN=$1
  local GPU=$2
  echo "[$(date '+%H:%M:%S')] PPO-teacher scenario $SCEN on cuda:$GPU START"
  pixi run sim test_ppo_v2 \
    test_ppo_v2.model_path=data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn/best_model/best_model.zip \
    test_ppo_v2.test_scenarios=[$SCEN] \
    "test_ppo_v2.test_seeds=[0,1,2,3,4]" \
    test_ppo_v2.n_episodes=100 \
    test_ppo_v2.init_episode=0 test_ppo_v2.max_episode=100 \
    test_ppo_v2.save_root=data/channel_generality/ppo_teacher_5seed \
    test_ppo_v2.save_metric_key=eval_ood \
    test_ppo_v2.device=cuda:$GPU
  echo "[$(date '+%H:%M:%S')] PPO-teacher scenario $SCEN DONE"
}

export -f run_e4_scenario run_ppo_scenario

echo "Functions exported. Use 'run_e4_scenario <SCEN> <GPU>' or 'run_ppo_scenario <SCEN> <GPU>'"
