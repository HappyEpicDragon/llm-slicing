#!/bin/bash
# S2: context_len 敏感性训练（4 个模型，各用一张 GPU 并行）
#
# 基准：E4 参数（D-B + SliceAttn, n_layer=6, n_head=8, embed_dim=512, epochs=60）
# 扫描：context_len ∈ [5, 10, 30, 50]（context_len=20 = E4，跳过）
#
# GPU 分配：
#   ctx5  → GPU 0
#   ctx10 → GPU 1
#   ctx30 → GPU 2
#   ctx50 → GPU 3
#
# 注意：batch_size=512 for ctx≤30，256 for ctx=50（序列更长，注意力内存约 6x）
set -e
cd /root/decision_transformer_slicing

PROJ=/root/decision_transformer_slicing
BASE=${PROJ}/data/channel_generality/dt_v2_8exp
DS=${BASE}/dataset_D-B
SAVE_BASE=${PROJ}/data/channel_generality/sensitivity/s2_context
LOG=${PROJ}/data/channel_generality/sensitivity/logs
mkdir -p ${LOG}

echo "=== S2 context_len 训练启动 $(date) ==="

# ctx5 → GPU 0
CUDA_VISIBLE_DEVICES=0 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DS} \
  train_dt_v2.save_dir=${SAVE_BASE}/ctx5 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=5 \
  train_dt_v2.model.embed_dim=512 \
  train_dt_v2.model.n_layer=6 \
  train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 \
  train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 \
  train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx5.log 2>&1 &
echo "  ctx5  → GPU 0 (pid $!)"

# ctx10 → GPU 1
CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DS} \
  train_dt_v2.save_dir=${SAVE_BASE}/ctx10 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=10 \
  train_dt_v2.model.embed_dim=512 \
  train_dt_v2.model.n_layer=6 \
  train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 \
  train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 \
  train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx10.log 2>&1 &
echo "  ctx10 → GPU 1 (pid $!)"

# ctx30 → GPU 2
CUDA_VISIBLE_DEVICES=2 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DS} \
  train_dt_v2.save_dir=${SAVE_BASE}/ctx30 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=30 \
  train_dt_v2.model.embed_dim=512 \
  train_dt_v2.model.n_layer=6 \
  train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 \
  train_dt_v2.optimizer.batch_size=512 \
  train_dt_v2.optimizer.lr=1e-4 \
  train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx30.log 2>&1 &
echo "  ctx30 → GPU 2 (pid $!)"

# ctx50 → GPU 3，batch_size 减半避免 OOM
CUDA_VISIBLE_DEVICES=3 /root/.pixi/bin/pixi run python -u main.py \
  'simulation=channel_generality/train_dt_v2' \
  train_dt_v2.dataset_dir=${DS} \
  train_dt_v2.save_dir=${SAVE_BASE}/ctx50 \
  train_dt_v2.model.encoder_type=slice_attn \
  train_dt_v2.model.context_len=50 \
  train_dt_v2.model.embed_dim=512 \
  train_dt_v2.model.n_layer=6 \
  train_dt_v2.model.n_head=8 \
  train_dt_v2.optimizer.epochs=60 \
  train_dt_v2.optimizer.batch_size=256 \
  train_dt_v2.optimizer.lr=1e-4 \
  train_dt_v2.device=cuda \
  > ${LOG}/s2_ctx50.log 2>&1 &
echo "  ctx50 → GPU 3 (pid $!, batch=256)"

echo ""
echo "全部后台启动完毕，等待完成..."
wait
echo "=== S2 训练全部完成 $(date) ==="
