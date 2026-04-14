#!/bin/bash
# S2/S3 敏感性分析完整流水线（修订版，直接启动，不依赖 tmux session 检测）
#
# 流程：
#   1. S2 训练（4 个 context_len × 4 GPU 并行）
#   2. S2 评测（4 GPU 并行）
#   3. S3 训练（3 个 dataset_size × 3 GPU 并行）
#   4. S3 评测（3 GPU 并行）
#   5. 生成 S2/S3 敏感性图
set -e
cd /root/decision_transformer_slicing

PROJ=/root/decision_transformer_slicing
DB=${PROJ}/data/channel_generality/dt_v2_8exp/dataset_D-B
META=${DB}/metadata.json
S2_BASE=${PROJ}/data/channel_generality/sensitivity/s2_context
S3_BASE=${PROJ}/data/channel_generality/sensitivity/s3_dataset
LOG=${PROJ}/data/channel_generality/sensitivity/logs
mkdir -p ${LOG}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Step 1: S2 训练 ───────────────────────────────────────────────────────
log "=== Step 1: S2 context_len 训练（4 GPU 并行）==="

CUDA_VISIBLE_DEVICES=0 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DB} \
  train_dt_v2.save_dir=${S2_BASE}/ctx5 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=5 \
  train_dt_v2.model.embed_dim=512 train_dt_v2.model.n_layer=6 train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx5.log 2>&1 &
PID_CTX5=$!
log "  ctx5  → GPU 0 (pid ${PID_CTX5})"

CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DB} \
  train_dt_v2.save_dir=${S2_BASE}/ctx10 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=10 \
  train_dt_v2.model.embed_dim=512 train_dt_v2.model.n_layer=6 train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx10.log 2>&1 &
PID_CTX10=$!
log "  ctx10 → GPU 1 (pid ${PID_CTX10})"

CUDA_VISIBLE_DEVICES=2 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DB} \
  train_dt_v2.save_dir=${S2_BASE}/ctx30 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=30 \
  train_dt_v2.model.embed_dim=512 train_dt_v2.model.n_layer=6 train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx30.log 2>&1 &
PID_CTX30=$!
log "  ctx30 → GPU 2 (pid ${PID_CTX30})"

CUDA_VISIBLE_DEVICES=3 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DB} \
  train_dt_v2.save_dir=${S2_BASE}/ctx50 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=50 \
  train_dt_v2.model.embed_dim=512 train_dt_v2.model.n_layer=6 train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 train_dt_v2.optimizer.batch_size=256 \
  train_dt_v2.optimizer.lr=1e-4 train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx50.log 2>&1 &
PID_CTX50=$!
log "  ctx50 → GPU 3 (pid ${PID_CTX50}, batch=256)"

log "等待 S2 训练完成..."
wait ${PID_CTX5} ${PID_CTX10} ${PID_CTX30} ${PID_CTX50}
log "=== S2 训练完成 ==="

# ─── Step 2: S2 评测 ───────────────────────────────────────────────────────
log "=== Step 2: S2 评测（4 GPU 并行）==="

for CTX in 5 10 30 50; do
  GPU=$((CTX == 5 ? 0 : (CTX == 10 ? 1 : (CTX == 30 ? 2 : 3))))
  MODEL=${S2_BASE}/ctx${CTX}/epoch_60.pth
  if [ ! -f "${MODEL}" ]; then
    log "  [SKIP] ctx${CTX}: ${MODEL} 不存在"
    continue
  fi
  CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run python -u main.py \
    'simulation=channel_generality/test_dt_v2' \
    test_dt_v2.model_path=${MODEL} \
    test_dt_v2.meta_path=${META} \
    test_dt_v2.encoder_type=slice_attn \
    test_dt_v2.model.context_len=${CTX} \
    test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
    'test_dt_v2.test_scenarios=[5,6,7,8,9]' 'test_dt_v2.test_seeds=[0]' \
    test_dt_v2.n_episodes=20 \
    test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
    test_dt_v2.target_rtg=0 \
    test_dt_v2.save_root=${S2_BASE}/ctx${CTX}/eval \
    test_dt_v2.device=cuda \
    > ${LOG}/s2_eval_ctx${CTX}.log 2>&1 &
  log "  ctx${CTX} eval → GPU ${GPU} (pid $!)"
done
wait
log "=== S2 评测完成 ==="

# ─── Step 3: S3 训练 ───────────────────────────────────────────────────────
log "=== Step 3: S3 dataset size 训练（3 GPU 并行）==="

for PCT in 10 30 50; do
  GPU=$((PCT == 10 ? 0 : (PCT == 30 ? 1 : 2)))
  CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run python -u main.py \
    'simulation=channel_generality/train_dt_v2' \
    train_dt_v2.dataset_dir=${S3_BASE}/dataset_${PCT}pct \
    train_dt_v2.save_dir=${S3_BASE}/pct${PCT} \
    train_dt_v2.model.encoder_type=slice_attn \
    train_dt_v2.model.context_len=20 \
    train_dt_v2.model.embed_dim=512 train_dt_v2.model.n_layer=6 train_dt_v2.model.n_head=8 \
    train_dt_v2.optimizer.epochs=60 train_dt_v2.optimizer.batch_size=512 \
    train_dt_v2.optimizer.lr=1e-4 train_dt_v2.device=cuda \
    > ${LOG}/s3_pct${PCT}.log 2>&1 &
  log "  pct${PCT} → GPU ${GPU} (pid $!)"
done
wait
log "=== S3 训练完成 ==="

# ─── Step 4: S3 评测 ───────────────────────────────────────────────────────
log "=== Step 4: S3 评测（3 GPU 并行）==="

for PCT in 10 30 50; do
  GPU=$((PCT == 10 ? 0 : (PCT == 30 ? 1 : 2)))
  MODEL=${S3_BASE}/pct${PCT}/epoch_60.pth
  if [ ! -f "${MODEL}" ]; then
    log "  [SKIP] pct${PCT}: ${MODEL} 不存在"
    continue
  fi
  CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run python -u main.py \
    'simulation=channel_generality/test_dt_v2' \
    test_dt_v2.model_path=${MODEL} \
    test_dt_v2.meta_path=${META} \
    test_dt_v2.encoder_type=slice_attn \
    test_dt_v2.model.context_len=20 \
    test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
    'test_dt_v2.test_scenarios=[5,6,7,8,9]' 'test_dt_v2.test_seeds=[0]' \
    test_dt_v2.n_episodes=20 \
    test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
    test_dt_v2.target_rtg=0 \
    test_dt_v2.save_root=${S3_BASE}/pct${PCT}/eval \
    test_dt_v2.device=cuda \
    > ${LOG}/s3_eval_pct${PCT}.log 2>&1 &
  log "  pct${PCT} eval → GPU ${GPU} (pid $!)"
done
wait
log "=== S3 评测完成 ==="

# ─── Step 5: 出图 ──────────────────────────────────────────────────────────
log "=== Step 5: 生成 S2/S3 图表 ==="
/root/.pixi/bin/pixi run python scripts/plot_s2_s3_sensitivity.py
log "=== S2/S3 流水线全部完成 ==="
