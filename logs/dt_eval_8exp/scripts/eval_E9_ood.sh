#!/usr/bin/env bash
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=3
echo "[E9] checkpoint=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dt_model_E9_mlp/epoch_30.pth"
pixi run sim test_dt_v2 \
  test_dt_v2.model_path=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dt_model_E9_mlp/epoch_30.pth \
  test_dt_v2.meta_path=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dataset_D-A/metadata.json \
  test_dt_v2.encoder_type=mlp \
  test_dt_v2.model.n_layer=6 \
  test_dt_v2.model.n_head=8 \
  test_dt_v2.model.embed_dim=512 \
  test_dt_v2.test_scenarios=[5,6,7,8,9] \
  test_dt_v2.test_seeds=[0] \
  test_dt_v2.n_episodes=20 \
  test_dt_v2.device=cuda \
  test_dt_v2.save_root=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/eval_ood/E9 \
  2>&1 | tee /root/decision_transformer_slicing/logs/dt_eval_8exp/E9_mlp_ood.log
echo "EXIT:$?"
