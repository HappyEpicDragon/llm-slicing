"""生成 10 个 DT 训练脚本（E1-E10）。"""
import os

PROJ = "/root/decision_transformer_slicing"
BASE = f"{PROJ}/data/channel_generality/dt_v2_8exp"
LOG  = f"{PROJ}/logs/dt_train_8exp"
SCRIPTS = f"{LOG}/scripts"
os.makedirs(SCRIPTS, exist_ok=True)

EXPERIMENTS = [
    # (exp_id, dataset, encoder, gpu)
    # 第一波：GPU 0-3 各跑1个
    ("E1",  "dataset_D-A", "mlp",        0),
    ("E2",  "dataset_D-A", "slice_attn", 1),
    ("E3",  "dataset_D-B", "mlp",        2),
    ("E4",  "dataset_D-B", "slice_attn", 3),
    # 第二波：E5-E8
    ("E5",  "dataset_D-C", "mlp",        0),
    ("E6",  "dataset_D-C", "slice_attn", 1),
    ("E7",  "dataset_D-E", "mlp",        2),
    ("E8",  "dataset_D-E", "slice_attn", 3),
    # 第三波：E9-E10（GPU 0-1）
    ("E9",  "dataset_D-D", "mlp",        0),
    ("E10", "dataset_D-D", "slice_attn", 1),
]

for exp_id, dataset, encoder, gpu in EXPERIMENTS:
    model_dir = f"{BASE}/dt_model_{exp_id}_{encoder}"
    script_path = os.path.join(SCRIPTS, f"train_{exp_id}.sh")
    with open(script_path, "w") as f:
        f.write(f"""#!/usr/bin/env bash
set -e
cd {PROJ}
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES={gpu}
echo "[{exp_id}] dataset={dataset} encoder={encoder} gpu={gpu}"
pixi run sim train_dt_v2 \\
  train_dt_v2.dataset_dir={BASE}/{dataset} \\
  train_dt_v2.save_dir={model_dir} \\
  train_dt_v2.model.encoder_type={encoder} \\
  train_dt_v2.model.embed_dim=512 \\
  train_dt_v2.model.n_layer=6 \\
  train_dt_v2.model.n_head=8 \\
  train_dt_v2.optimizer.epochs=30 \\
  train_dt_v2.optimizer.batch_size=512 \\
  train_dt_v2.optimizer.lr=1e-4 \\
  train_dt_v2.device=cuda \\
  2>&1 | tee {LOG}/{exp_id}_{encoder}.log
echo "EXIT:$?"
""")
    os.chmod(script_path, 0o755)
    print(f"Written: {exp_id} ({dataset}, {encoder}, gpu={gpu})")

print(f"\nAll 10 scripts in {SCRIPTS}/")
print("\nWave 1 (E1-E4):  tmux sessions dt_E1..dt_E4")
print("Wave 2 (E5-E8):  after wave 1 finishes")
print("Wave 3 (E9-E10): after wave 2 finishes")
