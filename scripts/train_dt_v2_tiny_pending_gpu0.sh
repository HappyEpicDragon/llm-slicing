#!/bin/bash
set -euo pipefail

cd /root/decision_transformer_slicing
LOG_DIR="data/channel_generality/dt_v2_tiny/logs"
mkdir -p "$LOG_DIR"

gpu="cuda:0"

echo "=== [GPU0] START b3_e3 ===" | tee -a "$LOG_DIR/b3_e3.log"
PYTHONPATH=. /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
  train_dt_v2_tiny.grid_id=b3_e3 \
  train_dt_v2_tiny.model.embed_dim=24 \
  train_dt_v2_tiny.model.n_layer=2 \
  train_dt_v2_tiny.model.n_head=2 \
  train_dt_v2_tiny.model.encoder_type=slice_attn \
  train_dt_v2_tiny.model.encoder_hidden_dim=16 \
  train_dt_v2_tiny.model.encoder_num_heads=2 \
  train_dt_v2_tiny.device="${gpu}" \
  >> "$LOG_DIR/b3_e3.log" 2>&1

PYTHONPATH=. /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
  test_dt_v2_tiny.grid_id=b3_e3 \
  test_dt_v2_tiny.model_path=data/channel_generality/dt_v2_tiny/models/b3_e3/final_dt_v2.pth \
  test_dt_v2_tiny.model.embed_dim=24 \
  test_dt_v2_tiny.model.n_layer=2 \
  test_dt_v2_tiny.model.n_head=2 \
  test_dt_v2_tiny.encoder_type=slice_attn \
  test_dt_v2_tiny.model.encoder_hidden_dim=16 \
  test_dt_v2_tiny.model.encoder_num_heads=2 \
  test_dt_v2_tiny.test_seeds=[0] \
  test_dt_v2_tiny.device="${gpu}" \
  >> "$LOG_DIR/b3_e3.log" 2>&1

echo "=== [GPU0] DONE b3_e3 ===" | tee -a "$LOG_DIR/b3_e3.log"
