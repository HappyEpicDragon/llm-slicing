#!/bin/bash
# Wave 2: MLP S4+Joint，SliceAttn S0+S1，4卡并行，n_envs=12
cd /root/decision_transformer_slicing
LOG=data/channel_generality/dt_v2_mean_rwd/logs
mkdir -p $LOG

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 4 --timesteps 400000 --ent_coef 0.05 --extractor mlp --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s4 --device cuda:0 \
  > $LOG/w2_s4_mlp.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 0 1 2 3 4 --timesteps 400000 --ent_coef 0.05 --extractor mlp --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_joint_mlp --device cuda:1 \
  > $LOG/w2_joint_mlp.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 0 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s0_slice_attn --device cuda:2 \
  > $LOG/w2_s0_attn.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 1 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn --n_envs 12 \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s1_slice_attn --device cuda:3 \
  > $LOG/w2_s1_attn.log 2>&1 &

echo "Wave 2 launched (pids: $(jobs -p))"
wait && echo "=== WAVE 2 DONE ==="
