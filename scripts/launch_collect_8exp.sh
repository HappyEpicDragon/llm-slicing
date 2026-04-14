#!/usr/bin/env bash
# 启动 8 组对比实验的数据采集
# 11 个并行 tmux 会话（全部 CPU bound）：
#   5 × Attn per-scenario expert  →  dt_v2_8exp/dataset_sources/attn_expert/
#   1 × Attn joint                →  dt_v2_8exp/dataset_sources/attn_joint/
#   5 × MLP per-scenario expert   →  dt_v2_8exp/dataset_sources/mlp_expert/

set -e
PROJ=/root/decision_transformer_slicing
BASE=$PROJ/data/channel_generality/dt_v2_mean_rwd
OUT=$PROJ/data/channel_generality/dt_v2_8exp/dataset_sources
LOG=$PROJ/logs/collect_8exp

# ── Attn per-scenario experts ──────────────────────────────────────────────
for S in 0 1 2 3 4; do
  SESSION="collect_attn_s${S}"
  tmux kill-session -t $SESSION 2>/dev/null || true
  tmux new-session -d -s $SESSION -x 220 -y 50
  tmux send-keys -t $SESSION "
cd $PROJ && PYTHONPATH=. /usr/bin/env bash -c '
  pixi run sim collect_data_v2 \
    collect_data_v2.model_path=${BASE}/ppo_s${S}_slice_attn/best_model/best_model.zip \
    collect_data_v2.teacher_kind=single \
    collect_data_v2.teacher_id=ppo_s${S}_slice_attn \
    collect_data_v2.scenarios=[${S}] \
    collect_data_v2.n_episodes=60 \
    collect_data_v2.epsilon=0.1 \
    collect_data_v2.seed=10 \
    collect_data_v2.parallel=5 \
    collect_data_v2.output=${OUT}/attn_expert \
    2>&1 | tee ${LOG}/attn_s${S}.log
'
" Enter
  echo "Launched: $SESSION"
done

# ── Attn joint ─────────────────────────────────────────────────────────────
SESSION="collect_attn_joint"
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -x 220 -y 50
tmux send-keys -t $SESSION "
cd $PROJ && PYTHONPATH=. /usr/bin/env bash -c '
  pixi run sim collect_data_v2 \
    collect_data_v2.model_path=${BASE}/ppo_joint_slice_attn/best_model/best_model.zip \
    collect_data_v2.teacher_kind=joint \
    collect_data_v2.teacher_id=ppo_joint_slice_attn \
    \"collect_data_v2.scenarios=[0,1,2,3,4]\" \
    collect_data_v2.n_episodes=60 \
    collect_data_v2.epsilon=0.1 \
    collect_data_v2.seed=10 \
    collect_data_v2.parallel=5 \
    collect_data_v2.output=${OUT}/attn_joint \
    2>&1 | tee ${LOG}/attn_joint.log
'
" Enter
echo "Launched: $SESSION"

# ── MLP joint ─────────────────────────────────────────────────────────────
SESSION="collect_mlp_joint"
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -x 220 -y 50
tmux send-keys -t $SESSION "
cd $PROJ && PYTHONPATH=. /usr/bin/env bash -c '
  pixi run sim collect_data_v2 \
    collect_data_v2.model_path=${BASE}/ppo_joint_mlp/best_model/best_model.zip \
    collect_data_v2.teacher_kind=joint \
    collect_data_v2.teacher_id=ppo_joint_mlp \
    \"collect_data_v2.scenarios=[0,1,2,3,4]\" \
    collect_data_v2.n_episodes=60 \
    collect_data_v2.epsilon=0.1 \
    collect_data_v2.seed=10 \
    collect_data_v2.parallel=5 \
    collect_data_v2.output=${OUT}/mlp_joint \
    2>&1 | tee ${LOG}/mlp_joint.log
'
" Enter
echo "Launched: $SESSION"

# ── MLP per-scenario experts ───────────────────────────────────────────────
for S in 0 1 2 3 4; do
  SESSION="collect_mlp_s${S}"
  tmux kill-session -t $SESSION 2>/dev/null || true
  tmux new-session -d -s $SESSION -x 220 -y 50
  tmux send-keys -t $SESSION "
cd $PROJ && PYTHONPATH=. /usr/bin/env bash -c '
  pixi run sim collect_data_v2 \
    collect_data_v2.model_path=${BASE}/ppo_s${S}/best_model/best_model.zip \
    collect_data_v2.teacher_kind=single \
    collect_data_v2.teacher_id=ppo_s${S}_mlp \
    collect_data_v2.scenarios=[${S}] \
    collect_data_v2.n_episodes=60 \
    collect_data_v2.epsilon=0.1 \
    collect_data_v2.seed=10 \
    collect_data_v2.parallel=5 \
    collect_data_v2.output=${OUT}/mlp_expert \
    2>&1 | tee ${LOG}/mlp_s${S}.log
'
" Enter
  echo "Launched: $SESSION"
done

echo ""
echo "All 11 collection sessions launched."
echo "Monitor:  tail -f $LOG/attn_s0.log"
echo "Check all: for s in 0 1 2 3 4; do echo -n \"attn_s\$s: \"; wc -l < $LOG/attn_s\$s.log 2>/dev/null || echo 0; done"
