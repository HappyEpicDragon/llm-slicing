#!/usr/bin/env bash
# 自动等待指定 GPU 空闲后启动 DT 续训任务。
# 用法：bash scripts/launch_dt_resume.sh
# 效果：
#   GPU2: E7 结束后 → resume_E3
#   GPU3: E8 结束后 → resume_E4
#   GPU0: E9_60ep 结束后 → resume_E1  (需先手动处理 E9)
#   GPU1: E10_60ep 结束后 → resume_E2  (需先手动处理 E10)

PROJ="/root/decision_transformer_slicing"
SCRIPTS="${PROJ}/logs/dt_train_8exp/scripts"

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

# --- GPU 2: E7 结束后 resume E3 ---
(
    wait_gpu_free 2
    echo "[launch] Starting resume_E3 on GPU 2"
    tmux new-session -d -s dt_E3r "bash ${SCRIPTS}/resume_E3.sh"
    echo "[launch] tmux session dt_E3r started"
) &

# --- GPU 3: E8 结束后 resume E4 ---
(
    wait_gpu_free 3
    echo "[launch] Starting resume_E4 on GPU 3"
    tmux new-session -d -s dt_E4r "bash ${SCRIPTS}/resume_E4.sh"
    echo "[launch] tmux session dt_E4r started"
) &

echo "[launcher] Background watchers started for GPU2 (E3) and GPU3 (E4)."
echo "[launcher] Note: E9/E10 are still on the old 30ep chain in dt_E1/dt_E2."
echo "[launcher] ACTION REQUIRED when E5/E6 finish:"
echo "  1) Kill dt_E1: tmux kill-session -t dt_E1"
echo "  2) Kill dt_E2: tmux kill-session -t dt_E2"
echo "  3) Launch E9 60ep: tmux new-session -d -s dt_E9 \"bash ${SCRIPTS}/train_E9_60ep.sh\""
echo "  4) Launch E10 60ep: tmux new-session -d -s dt_E10 \"bash ${SCRIPTS}/train_E10_60ep.sh\""
echo "  5) After E9/E10 finish, run resume_E1 and resume_E2."

wait
