#!/bin/bash
# S1: target_rtg 敏感性扫描
#
# 扫描范围依据：训练数据 episode return ∈ [-199, 0]，经 sign*log1p 变换后
# 模型实际输入范围约为 [-5.3, 0]。
#
# 旧范围 [0,100,...,50000] 全部为正值（log-val > 0），均在训练分布之外，无意义。
# 新范围：[-200,-100,-50,-20,-10,-5,0,10,50]
#   - [-200, 0]：覆盖"最差到最优"的完整训练分布
#   - [10, 50]：两个正值外推点，用于观察 OOD RTG 的影响
#
# 注意：pixi run sim <mode> 会自动加 channel_generality/ 前缀，勿重复。
set -e
export CUDA_VISIBLE_DEVICES=0
cd /root/decision_transformer_slicing

for RTG in -200 -100 -50 -20 -10 -5 0 10 50; do
    echo "=== RTG=$RTG starting at $(date) ==="
    SAVE_DIR="data/channel_generality/dt_v2_8exp/sensitivity/rtg_${RTG}"
    pixi run sim test_dt_v2 \
      test_dt_v2.model_path=data/channel_generality/dt_v2_8exp/dt_model_E4_slice_attn/epoch_60.pth \
      test_dt_v2.meta_path=data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json \
      test_dt_v2.encoder_type=slice_attn \
      test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
      'test_dt_v2.test_scenarios=[5,6,7,8,9]' \
      'test_dt_v2.test_seeds=[0]' \
      test_dt_v2.n_episodes=20 \
      test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
      "test_dt_v2.target_rtg=${RTG}" \
      "test_dt_v2.save_root=${SAVE_DIR}" \
      test_dt_v2.device=cuda
    echo "=== RTG=$RTG done at $(date) ==="
done
echo "=== ALL RTG SWEEPS COMPLETE ==="
