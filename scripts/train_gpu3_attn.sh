#!/bin/bash
cd /root/decision_transformer_slicing
LOG=data/channel_generality/dt_v2_mean_rwd/logs
mkdir -p $LOG

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 3 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s3_slice_attn --device cuda:3 \
  > $LOG/gpu3_s3.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 4 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_s4_slice_attn --device cuda:3 \
  > $LOG/gpu3_s4.log 2>&1 &

PYTHONPATH=. /root/.pixi/bin/pixi run python -u src/basic_apis/dt_v2/train_ppo_v2.py \
  --scenarios 0 1 2 3 4 --timesteps 400000 --ent_coef 0.05 --extractor slice_attn \
  --save_dir data/channel_generality/dt_v2_mean_rwd/ppo_joint_slice_attn --device cuda:3 \
  > $LOG/gpu3_joint.log 2>&1 &

echo "GPU3 SliceAttn S3+S4+Joint launched (pids: $(jobs -p))"
wait
echo "=== GPU3 SliceAttn S3-S4-Joint ALL DONE ==="
