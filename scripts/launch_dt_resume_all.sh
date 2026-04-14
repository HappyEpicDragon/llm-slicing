#!/usr/bin/env bash
# 方案 A 完整续训调度：所有 10 个模型均续训至 60 epoch
# GPU0: [E5运行中] → E9(30ep,自动) → E1r(ep31-60) → E5r(ep31-60)
# GPU1: [E6运行中] → E10(30ep,自动) → E2r(ep31-60) → E6r(ep31-60)
# GPU2: [E7运行中] → E3r(ep31-60) → E7r(ep31-60)  ← E3r/E4r已由watcher接管
# GPU3: [E8运行中] → E4r(ep31-60) → E8r(ep31-60)
#
# 本脚本负责 GPU2/GPU3 的第二段（E7r/E8r）和 GPU0/GPU1 的全链（E1r/E2r/E5r/E6r）

PROJ="/root/decision_transformer_slicing"
SCRIPTS="${PROJ}/logs/dt_train_8exp/scripts"
BASE="${PROJ}/data/channel_generality/dt_v2_8exp"
LOG="${PROJ}/logs/dt_train_8exp"

wait_file() {
    local f="$1"
    echo "[wait] Waiting for file: $f"
    while [ ! -f "$f" ]; do sleep 60; done
    echo "[wait] File found: $f"
    sleep 10   # 等写完
}

wait_gpu_free() {
    local gpu_id=$1
    echo "[wait] Waiting for GPU ${gpu_id} to be free..."
    while true; do
        pids=$(nvidia-smi -i ${gpu_id} --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l)
        if [ "$pids" -eq 0 ]; then
            echo "[wait] GPU ${gpu_id} is free."
            return 0
        fi
        sleep 30
    done
}

run_in_tmux() {
    local session="$1"
    local script="$2"
    echo "[launch] tmux new-session -d -s ${session} ..."
    tmux new-session -d -s "${session}" "bash ${script}; echo SESSION_DONE"
}

# ===== GPU 2: E3r 完成后 → E7r =====
(
    wait_file "${BASE}/dt_model_E3_mlp/epoch_60.pth"
    echo "[GPU2] E3r done. Starting E7r..."
    run_in_tmux "dt_E7r" "${SCRIPTS}/resume_E7.sh"
) &

# ===== GPU 3: E4r 完成后 → E8r =====
(
    wait_file "${BASE}/dt_model_E4_slice_attn/epoch_60.pth"
    echo "[GPU3] E4r done. Starting E8r..."
    run_in_tmux "dt_E8r" "${SCRIPTS}/resume_E8.sh"
) &

# ===== GPU 0: E9(30ep) 完成后 → E1r → E5r =====
(
    wait_file "${BASE}/dt_model_E9_mlp/epoch_30.pth"
    echo "[GPU0] E9 done. Starting E1r..."
    run_in_tmux "dt_E1r" "${SCRIPTS}/resume_E1.sh"
    wait_file "${BASE}/dt_model_E1_mlp/epoch_60.pth"
    echo "[GPU0] E1r done. Starting E5r..."
    run_in_tmux "dt_E5r" "${SCRIPTS}/resume_E5.sh"
) &

# ===== GPU 1: E10(30ep) 完成后 → E2r → E6r =====
(
    wait_file "${BASE}/dt_model_E10_slice_attn/epoch_30.pth"
    echo "[GPU1] E10 done. Starting E2r..."
    run_in_tmux "dt_E2r" "${SCRIPTS}/resume_E2.sh"
    wait_file "${BASE}/dt_model_E2_slice_attn/epoch_60.pth"
    echo "[GPU1] E2r done. Starting E6r..."
    run_in_tmux "dt_E6r" "${SCRIPTS}/resume_E6.sh"
) &

echo "[all_scheduler] All 4 GPU chains are being watched."
echo "[all_scheduler] Expected flow:"
echo "  GPU0: E5→E9(30ep)→E1r(ep31-60)→E5r(ep31-60)"
echo "  GPU1: E6→E10(30ep)→E2r(ep31-60)→E6r(ep31-60)"
echo "  GPU2: E7→E3r(ep31-60)→E7r(ep31-60)"
echo "  GPU3: E8→E4r(ep31-60)→E8r(ep31-60)"
echo "  (E3r/E4r startup already handled by dt_resume_watcher)"
echo ""
echo "  Total wall time estimate (from now):"
echo "    GPU2/GPU3: ~1.8h(E7/E8) + 2h(E3r/E4r) + 2h(E7r/E8r) ≈ 6h"
echo "    GPU0/GPU1: ~1.8h(E5/E6) + 8h(E9/E10) + 2h(E1r/E2r) + 4h(E5r/E6r) ≈ 16h"

wait
