#!/bin/bash
# Wave 1: MLP S0-S3，4卡并行，n_envs=12
cd /root/decision_transformer_slicing
LOG=data/channel_generality/dt_v2_mean_rwd/logs
mkdir -p $LOG

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 0 --timesteps 400000 --ent_coef 0.05 --extractor mlp --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s0 --device cuda:0 \
  > $LOG/w1_s0_mlp.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 1 --timesteps 400000 --ent_coef 0.05 --extractor mlp --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s1 --device cuda:1 \
  > $LOG/w1_s1_mlp.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 2 --timesteps 400000 --ent_coef 0.05 --extractor mlp --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s2 --device cuda:2 \
  > $LOG/w1_s2_mlp.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 3 --timesteps 400000 --ent_coef 0.05 --extractor mlp --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s3 --device cuda:3 \
  > $LOG/w1_s3_mlp.log 2>&1 &

echo "Wave 1 launched (pids: $(jobs -p))"
wait && echo "=== WAVE 1 DONE ==="
