#!/usr/bin/env bash
set -e
cd /root/decision_transformer_slicing
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=2
echo "[E7] dataset=dataset_D-E encoder=mlp gpu=2"
pixi run sim train_dt_v2 \
  train_dt_v2.dataset_dir=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dataset_D-E \
  train_dt_v2.save_dir=/root/decision_transformer_slicing/data/channel_generality/dt_v2_8exp/dt_model_E7_mlp \
  train_dt_v2.model.encoder_type=mlp \
  train_dt_v2.model.embed_dim=512 \
  train_dt_v2.model.n_layer=6 \
  train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=30 \
  train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 \
  train_dt_v2.device=cuda \
  2>&1 | tee /root/decision_transformer_slicing/logs/dt_train_8exp/E7_mlp.log
echo "EXIT:$?"
