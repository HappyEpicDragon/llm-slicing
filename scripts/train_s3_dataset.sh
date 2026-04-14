#!/bin/bash
# S3: dataset size 敏感性训练（3 个模型，各用一张 GPU 并行）
#
# 基准：E4 参数（D-B 100% + SliceAttn, n_layer=6, n_head=8, context_len=20, epochs=60）
# 扫描：dataset_size ∈ [10%, 30%, 50%]（100% = E4，跳过）
#
# GPU 分配：
#   pct10 → GPU 0
#   pct30 → GPU 1
#   pct50 → GPU 2
set -e
cd /root/decision_transformer_slicing

PROJ=/root/decision_transformer_slicing
DS_BASE=${PROJ}/data/channel_generality/sensitivity/s3_dataset
SAVE_BASE=${PROJ}/data/channel_generality/sensitivity/s3_dataset
LOG=${PROJ}/data/channel_generality/sensitivity/logs
mkdir -p ${LOG}

echo "=== S3 dataset size 训练启动 $(date) ==="

for PCT in 10 30 50; do
  GPU=$((PCT == 10 ? 0 : (PCT == 30 ? 1 : 2)))
  CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run python -u main.py \
    'simulation=channel_generality/train_dt_v2' \
    train_dt_v2.dataset_dir=${DS_BASE}/dataset_${PCT}pct \
    train_dt_v2.save_dir=${SAVE_BASE}/pct${PCT} \
    train_dt_v2.model.encoder_type=slice_attn \
    train_dt_v2.model.context_len=20 \
    train_dt_v2.model.embed_dim=512 \
    train_dt_v2.model.n_layer=6 \
    train_dt_v2.model.n_head=8 \
    train_dt_v2.optimizer.epochs=60 \
    train_dt_v2.optimizer.batch_size=512 \
    train_dt_v2.optimizer.lr=1e-4 \
    train_dt_v2.device=cuda \
    > ${LOG}/s3_pct${PCT}.log 2>&1 &
  echo "  pct${PCT} → GPU ${GPU} (pid $!)"
done

echo ""
echo "全部后台启动完毕，等待完成..."
wait
echo "=== S3 训练全部完成 $(date) ==="
