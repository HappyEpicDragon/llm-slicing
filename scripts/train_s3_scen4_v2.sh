#!/bin/bash
set -euo pipefail
cd /root/decision_transformer_slicing

BASE="data/channel_generality/sensitivity/s3_coverage/scen4_v2"
GPU=1
LOG="data/channel_generality/sensitivity/logs/s3_scen4_v2.log"
mkdir -p "$(dirname "${LOG}")"
: > "${LOG}"

echo "=== train scen4_v2 {s0,s1,s2,s4} gpu=${GPU} start $(date) ===" | tee -a "${LOG}"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir="${BASE}" \
    train_dt_v2_tiny.save_dir="${BASE}/model" \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda >> "${LOG}" 2>&1

echo "=== eval scen4_v2 gpu=${GPU} start $(date) ===" | tee -a "${LOG}"
PYTHONPATH=. CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path="${BASE}/model/final_dt_v2.pth" \
    test_dt_v2_tiny.meta_path="${BASE}/metadata.json" \
    test_dt_v2_tiny.model.context_len=20 \
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
    test_dt_v2_tiny.save_root="${BASE}/eval" \
    test_dt_v2_tiny.device=cuda >> "${LOG}" 2>&1

echo "=== scen4_v2 ALL DONE $(date) ===" | tee -a "${LOG}"
