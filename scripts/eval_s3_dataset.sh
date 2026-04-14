#!/bin/bash
# S3: dataset size 敏感性评测
# 评测 pct10/pct30/pct50 + E4(pct100) 在 s5-s9 上的表现
# 依赖：train_s3_dataset.sh 已完成
set -e
cd /root/decision_transformer_slicing

PROJ=/root/decision_transformer_slicing
BASE=${PROJ}/data/channel_generality
SAVE_BASE=${BASE}/sensitivity/s3_dataset
META=${BASE}/dt_v2_8exp/dataset_D-B/metadata.json
LOG=${BASE}/sensitivity/logs
mkdir -p ${LOG}

echo "=== S3 评测启动 $(date) ==="

for PCT in 10 30 50; do
  GPU=$((PCT == 10 ? 0 : (PCT == 30 ? 1 : 2)))
  MODEL=${SAVE_BASE}/pct${PCT}/epoch_60.pth
  if [ ! -f "${MODEL}" ]; then
    echo "  [SKIP] pct${PCT}: ${MODEL} 不存在，跳过"
    continue
  fi
  CUDA_VISIBLE_DEVICES=${GPU} /root/.pixi/bin/pixi run python -u main.py \
    'simulation=channel_generality/test_dt_v2' \
    test_dt_v2.model_path=${MODEL} \
    test_dt_v2.meta_path=${META} \
    test_dt_v2.encoder_type=slice_attn \
    test_dt_v2.model.context_len=20 \
    test_dt_v2.model.n_layer=6 \
    test_dt_v2.model.n_head=8 \
    'test_dt_v2.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2.test_seeds=[0]' \
    test_dt_v2.n_episodes=20 \
    test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
    test_dt_v2.target_rtg=0 \
    test_dt_v2.save_root=${SAVE_BASE}/pct${PCT}/eval \
    test_dt_v2.device=cuda \
    > ${LOG}/s3_eval_pct${PCT}.log 2>&1 &
  echo "  pct${PCT} eval → GPU ${GPU} (pid $!)"
done

echo "等待评测完成..."
wait
echo "=== S3 评测全部完成 $(date) ==="
