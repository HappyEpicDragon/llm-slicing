#!/bin/bash
set -euo pipefail

cd /root/decision_transformer_slicing
LOG_DIR="data/channel_generality/dt_v2_tiny/logs"
mkdir -p "$LOG_DIR"

run_one() {
  local grid_id="$1"
  local embed_dim="$2"
  local n_layer="$3"
  local n_head="$4"
  local enc_hidden="$5"
  local gpu="cuda:1"

  local log_file="$LOG_DIR/${grid_id}.log"
  echo "=== [GPU1] START ${grid_id} ===" | tee -a "$log_file"

  PYTHONPATH=. /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id="${grid_id}" \
    train_dt_v2_tiny.model.embed_dim="${embed_dim}" \
    train_dt_v2_tiny.model.n_layer="${n_layer}" \
    train_dt_v2_tiny.model.n_head="${n_head}" \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim="${enc_hidden}" \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.device="${gpu}" \
    >> "$log_file" 2>&1

  PYTHONPATH=. /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id="${grid_id}" \
    test_dt_v2_tiny.model_path="data/channel_generality/dt_v2_tiny/models/${grid_id}/final_dt_v2.pth" \
    test_dt_v2_tiny.model.embed_dim="${embed_dim}" \
    test_dt_v2_tiny.model.n_layer="${n_layer}" \
    test_dt_v2_tiny.model.n_head="${n_head}" \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim="${enc_hidden}" \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    test_dt_v2_tiny.device="${gpu}" \
    >> "$log_file" 2>&1

  echo "=== [GPU1] DONE ${grid_id} ===" | tee -a "$log_file"
}

run_one b1_e2 64 2 2 32
run_one b2_e3 32 2 2 16

echo "=== GPU1 ALL DONE ==="
