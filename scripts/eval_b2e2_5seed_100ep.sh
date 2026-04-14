#!/bin/bash
set -euo pipefail

cd /root/decision_transformer_slicing

LOG_DIR="data/channel_generality/dt_v2_tiny/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/b2_e2_eval_5seed_100ep.log"

echo "=== START b2_e2 5seed 100ep eval ===" | tee -a "$LOG_FILE"

PYTHONPATH=. /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
  test_dt_v2_tiny.grid_id=b2_e2 \
  test_dt_v2_tiny.model_path=data/channel_generality/dt_v2_tiny/models/b2_e2/final_dt_v2.pth \
  test_dt_v2_tiny.model.embed_dim=32 \
  test_dt_v2_tiny.model.n_layer=2 \
  test_dt_v2_tiny.model.n_head=2 \
  test_dt_v2_tiny.encoder_type=slice_attn \
  test_dt_v2_tiny.model.encoder_hidden_dim=32 \
  test_dt_v2_tiny.model.encoder_num_heads=2 \
  'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
  'test_dt_v2_tiny.test_seeds=[0,1,2,3,4]' \
  test_dt_v2_tiny.n_episodes=100 \
  test_dt_v2_tiny.init_episode=0 \
  test_dt_v2_tiny.max_episode=100 \
  test_dt_v2_tiny.save_root=data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100 \
  test_dt_v2_tiny.device=cuda:1 \
  >> "$LOG_FILE" 2>&1

echo "=== DONE b2_e2 5seed 100ep eval ===" | tee -a "$LOG_FILE"
