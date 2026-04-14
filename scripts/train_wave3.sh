#!/bin/bash
# Wave 3: SliceAttn S2-S4+Joint，4卡并行，n_envs=12
cd /root/decision_transformer_slicing
LOG=data/channel_generality/dt_v2_mean_rwd/logs
mkdir -p $LOG

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 2 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s2_slice_attn --device cuda:0 \
  > $LOG/w3_s2_attn.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 3 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s3_slice_attn --device cuda:1 \
  > $LOG/w3_s3_attn.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 4 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s4_slice_attn --device cuda:2 \
  > $LOG/w3_s4_attn.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 0 1 2 3 4 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn --device cuda:3 \
  > $LOG/w3_joint_attn.log 2>&1 &

echo "Wave 3 launched (pids: $(jobs -p))"
wait && echo "=== WAVE 3 DONE ==="
