#!/bin/bash
cd /root/decision_transformer_slicing
LOG=data/channel_generality/dt_v2_mean_rwd/logs
mkdir -p $LOG

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 0 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s0_slice_attn --device cuda:2 \
  > $LOG/gpu2_s0.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 1 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s1_slice_attn --device cuda:2 \
  > $LOG/gpu2_s1.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 2 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s2_slice_attn --device cuda:2 \
  > $LOG/gpu2_s2.log 2>&1 &

echo "GPU2 SliceAttn S0+S1+S2 launched (pids: $(jobs -p))"
wait
echo "=== GPU2 SliceAttn S0-S2 ALL DONE ==="
