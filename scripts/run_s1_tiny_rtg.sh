#!/bin/bash
set -euo pipefail

cd /root/decision_transformer_slicing

LOG_DIR="data/channel_generality/sensitivity/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/s1_rtg_tiny.log"
: > "${LOG_FILE}"

echo "=== S1 tiny RTG sweep start $(date) ===" | tee -a "${LOG_FILE}"

for RTG in -200 -100 -50 -20 -10 -5 0 10 50; do
  echo "=== RTG=${RTG} start $(date) ===" | tee -a "${LOG_FILE}"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/dt_v2_tiny/models/b2_e2/final_dt_v2.pth \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg="${RTG}" \
    test_dt_v2_tiny.save_root=data/channel_generality/dt_v2_tiny/sensitivity/rtg_"${RTG}" \
    test_dt_v2_tiny.device=cuda:0 >> "${LOG_FILE}" 2>&1
  echo "=== RTG=${RTG} done $(date) ===" | tee -a "${LOG_FILE}"
done

echo "=== S1 tiny RTG sweep done $(date) ===" | tee -a "${LOG_FILE}"
