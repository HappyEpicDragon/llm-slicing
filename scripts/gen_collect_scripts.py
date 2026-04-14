"""
生成 12 个数据采集脚本并写入 logs/collect_8exp/scripts/
每个脚本对应一个 teacher 模型，全部采用 n_episodes=180（在 ep 0-59 内三轮回绕）。
"""
import os

PROJ = "/root/decision_transformer_slicing"
BASE = f"{PROJ}/data/channel_generality/dt_v2_mean_rwd"
OUT  = f"{PROJ}/data/channel_generality/dt_v2_8exp/dataset_sources"
LOG  = f"{PROJ}/logs/collect_8exp"
SCRIPTS = f"{LOG}/scripts"

os.makedirs(SCRIPTS, exist_ok=True)
os.makedirs(f"{OUT}/mlp_joint/training", exist_ok=True)

TASKS = []

# 5 × Attn per-scenario expert
for s in range(5):
    TASKS.append(dict(
        name=f"collect_attn_s{s}",
        model=f"{BASE}/ppo_s{s}_slice_attn/best_model/best_model.zip",
        kind="single",
        tid=f"ppo_s{s}_slice_attn",
        scenarios=f"[{s}]",
        output=f"{OUT}/attn_expert",
        log=f"{LOG}/attn_s{s}.log",
    ))

# 1 × Attn joint
TASKS.append(dict(
    name="collect_attn_joint",
    model=f"{BASE}/ppo_joint_slice_attn/best_model/best_model.zip",
    kind="joint",
    tid="ppo_joint_slice_attn",
    scenarios="[0,1,2,3,4]",
    output=f"{OUT}/attn_joint",
    log=f"{LOG}/attn_joint.log",
))

# 5 × MLP per-scenario expert
for s in range(5):
    TASKS.append(dict(
        name=f"collect_mlp_s{s}",
        model=f"{BASE}/ppo_s{s}/best_model/best_model.zip",
        kind="single",
        tid=f"ppo_s{s}_mlp",
        scenarios=f"[{s}]",
        output=f"{OUT}/mlp_expert",
        log=f"{LOG}/mlp_s{s}.log",
    ))

# 1 × MLP joint
TASKS.append(dict(
    name="collect_mlp_joint",
    model=f"{BASE}/ppo_joint_mlp/best_model/best_model.zip",
    kind="joint",
    tid="ppo_joint_mlp",
    scenarios="[0,1,2,3,4]",
    output=f"{OUT}/mlp_joint",
    log=f"{LOG}/mlp_joint.log",
))

assert len(TASKS) == 12, f"Expected 12 tasks, got {len(TASKS)}"

for t in TASKS:
    script_path = os.path.join(SCRIPTS, f"{t['name']}.sh")
    with open(script_path, "w") as f:
        f.write(f"""#!/usr/bin/env bash
set -e
cd {PROJ}
export PYTHONPATH=.
pixi run sim collect_data_v2 \\
  collect_data_v2.model_path={t['model']} \\
  collect_data_v2.teacher_kind={t['kind']} \\
  collect_data_v2.teacher_id={t['tid']} \\
  'collect_data_v2.scenarios={t['scenarios']}' \\
  collect_data_v2.n_episodes=180 \\
  collect_data_v2.epsilon=0.1 \\
  collect_data_v2.seed=10 \\
  collect_data_v2.parallel=5 \\
  collect_data_v2.output={t['output']} \\
  2>&1 | tee {t['log']}
echo "EXIT:$?"
""")
    os.chmod(script_path, 0o755)
    print(f"Written: {script_path}")

print(f"\nAll {len(TASKS)} scripts generated in {SCRIPTS}/")
