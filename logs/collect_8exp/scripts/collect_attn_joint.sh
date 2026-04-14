#!/usr/bin/env bash
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.
pixi run sim collect_data_v2 \
  collect_data_v2.model_path=/root/decision_transformer_slicing/data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn/best_model/best_model.zip \
  collect_data_v2.teacher_kind=joint \
  collect_data_v2.teacher_id=ppo_joint_slice_attn \
  'collect_data_v2.scenarios=[0,1,2,3,4]' \
  collect_data_v2.n_episodes=180 \
  collect_data_v2.epsilon=0.1 \
  collect_data_v2.seed=10 \
  collect_data_v2.parallel=5 \
  collect_data_v2.output=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dataset_sources/attn_joint \
  2>&1 | tee /root/decision_transformer_slicing/logs/collect_8exp/attn_joint.log
echo "EXIT:$?"
