#!/usr/bin/env bash
# 补全 eval_ood_100ep/E4：在 s5–s9 上评测 seeds 1–4（100 ep, init/max 0/100），写入同一 save_root。
set -euo pipefail
cd /root/decision_transformer_slicing
export PYTHONPATH=.

SAVE_ROOT=data/channel_generality/dt_v2_8exp/eval_ood_100ep/E4
MODEL=data/channel_generality/dt_v2_8exp/dt_model_E4_slice_attn/epoch_60.pth
META=data/channel_generality/dt_v2_8exp/dataset_D-B/metadata.json
LOG_DIR=data/channel_generality/dt_v2_8exp/eval_ood_100ep/logs
mkdir -p "$LOG_DIR"

run_one() {
  local SCEN=$1 GPU=$2
  echo "[$(date +%H:%M:%S)] scenario $SCEN on cuda:$GPU start"
  pixi run sim test_dt_v2 \
    test_dt_v2.model_path="$MODEL" \
    test_dt_v2.meta_path="$META" \
    test_dt_v2.encoder_type=slice_attn \
    test_dt_v2.model.n_layer=6 test_dt_v2.model.n_head=8 \
    "test_dt_v2.test_scenarios=[$SCEN]" \
    "test_dt_v2.test_seeds=[1,2,3,4]" \
    test_dt_v2.n_episodes=100 \
    test_dt_v2.init_episode=0 test_dt_v2.max_episode=100 \
    test_dt_v2.save_root="$SAVE_ROOT" \
    test_dt_v2.device=cuda:$GPU \
    >"$LOG_DIR/scen${SCEN}_seeds1_4.log" 2>&1
  echo "[$(date +%H:%M:%S)] scenario $SCEN on cuda:$GPU done"
}

run_one 5 0 &
run_one 6 1 &
run_one 7 2 &
run_one 8 3 &
wait
run_one 9 0
echo "All scenarios done."
