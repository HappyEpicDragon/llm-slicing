#!/bin/bash
set -euo pipefail

cd /root/decision_transformer_slicing

LOG_DIR="data/channel_generality/sensitivity/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/s2_tiny.log"
: > "${LOG_FILE}"

echo "=== S2 tiny context sweep start $(date) ===" | tee -a "${LOG_FILE}"

for CTX in 5 10 30 50; do
  if [[ "${CTX}" == "5" || "${CTX}" == "30" ]]; then
    GPU=1
  else
    GPU=2
  fi

  SAVE_DIR="data/channel_generality/sensitivity/s2_context/ctx${CTX}"
  echo "=== train ctx=${CTX} gpu=${GPU} start $(date) ===" | tee -a "${LOG_FILE}"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/dt_v2_8exp/dataset_D-B \
    train_dt_v2_tiny.save_dir="${SAVE_DIR}" \
    train_dt_v2_tiny.model.context_len="${CTX}" \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda >> "${LOG_FILE}" 2>&1

  echo "=== eval ctx=${CTX} gpu=${GPU} start $(date) ===" | tee -a "${LOG_FILE}"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path="${SAVE_DIR}/final_dt_v2.pth" \
    test_dt_v2_tiny.meta_path=data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json \
    test_dt_v2_tiny.model.context_len="${CTX}" \
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
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root="${SAVE_DIR}/eval" \
    test_dt_v2_tiny.device=cuda >> "${LOG_FILE}" 2>&1

  echo "=== ctx=${CTX} done $(date) ===" | tee -a "${LOG_FILE}"
done

echo "=== S2 tiny context sweep done $(date) ===" | tee -a "${LOG_FILE}"
