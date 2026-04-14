#!/bin/bash
# Dim0 Sensitivity Sweep: 测试不同 codebook 条目对 DT 性能的影响
#
# 用法: cd /root/decision_transformer_slicing && bash scripts/sweep_dim0.sh
#
# 依次测试 force_dim0 = null(DT原始), 0, 1, 3, 5, 7, 10
# 结果保存在 data/channel_generality/dt_dim0_sweep/{label}/metric_json/

set -e
cd /root/decision_transformer_slicing

PIXI_PYTHON=".pixi/envs/default/bin/python"
BASE_SAVE="/root/decision_transformer_slicing/data/channel_generality/dt_dim0_sweep"

# 要测试的 Dim0 值
DIM0_VALUES=("null" "0" "1" "3" "5" "7" "10")

for dim0 in "${DIM0_VALUES[@]}"; do
    if [ "$dim0" = "null" ]; then
        label="dt_original"
        override="dt_testing.force_dim0=null"
    else
        label="act${dim0}"
        override="dt_testing.force_dim0=${dim0}"
    fi

    save_root="${BASE_SAVE}/${label}"
    echo ""
    echo "=============================================="
    echo "  Testing Dim0 = ${label}"
    echo "  Save to: ${save_root}/metric_json/"
    echo "=============================================="

    $PIXI_PYTHON main.py \
        simulation=channel_generality \
        simulation/channel_generality=dt_testing \
        "${override}" \
        "dt_testing.save_root=${save_root}" \
        "dt_testing.save_results=true" \
        "dt_testing.clean_before_save=true" \
        2>&1 | tail -30

    echo "  Done: ${label}"
done

echo ""
echo "=============================================="
echo "  Sweep Complete! Comparing results..."
echo "=============================================="

$PIXI_PYTHON scripts/compare_dim0_sweep.py
