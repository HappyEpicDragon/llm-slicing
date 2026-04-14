#!/usr/bin/env bash
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.
pixi run sim collect_data_v2 \
  collect_data_v2.model_path=/root/decision_transformer_slicing/data/channel_generality/dt_v2_mean_rwd/ppo_s3/best_model/best_model.zip \
  collect_data_v2.teacher_kind=single \
  collect_data_v2.teacher_id=ppo_s3_mlp \
  'collect_data_v2.scenarios=[3]' \
  collect_data_v2.n_episodes=180 \
  collect_data_v2.epsilon=0.1 \
  collect_data_v2.seed=10 \
  collect_data_v2.parallel=5 \
  collect_data_v2.output=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dataset_sources/mlp_expert \
  2>&1 | tee /root/decision_transformer_slicing/logs/collect_8exp/mlp_s3.log
echo "EXIT:$?"
