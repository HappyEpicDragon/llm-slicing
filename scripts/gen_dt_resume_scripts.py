"""生成 DT 续训脚本。
E1-E4 从 epoch_30.pth 续训 30 epoch（总共 60 epoch）。
E5-E8 从 epoch_30.pth 续训 30 epoch（总共 60 epoch）。
E9/E10 直接用 60 epoch（不续训，重新训练）。
"""
import os

PROJ = "/root/decision_transformer_slicing"
BASE = f"{PROJ}/data/channel_generality/dt_v2_8exp"
LOG  = f"{PROJ}/logs/dt_train_8exp"
SCRIPTS = f"{LOG}/scripts"
os.makedirs(SCRIPTS, exist_ok=True)

# 续训配置：(exp_id, encoder, gpu, resume_epoch)
RESUME_EXPS = [
    # E1-E4 从 epoch_30 续训（GPU 在 E5-E8/E7-E8 完成后释放）
    ("E1",  "mlp",        0, 30),
    ("E2",  "slice_attn", 1, 30),
    ("E3",  "mlp",        2, 30),
    ("E4",  "slice_attn", 3, 30),
    # E5-E8 从 epoch_30 续训
    ("E5",  "mlp",        0, 30),
    ("E6",  "slice_attn", 1, 30),
    ("E7",  "mlp",        2, 30),
    ("E8",  "slice_attn", 3, 30),
]

DATASET_MAP = {
    "E1": "dataset_D-A", "E2": "dataset_D-A",
    "E3": "dataset_D-B", "E4": "dataset_D-B",
    "E5": "dataset_D-C", "E6": "dataset_D-C",
    "E7": "dataset_D-E", "E8": "dataset_D-E",
}

for exp_id, encoder, gpu, resume_epoch in RESUME_EXPS:
    dataset = DATASET_MAP[exp_id]
    model_dir = f"{BASE}/dt_model_{exp_id}_{encoder}"
    resume_ckpt = f"{model_dir}/epoch_{resume_epoch}.pth"
    script_path = os.path.join(SCRIPTS, f"resume_{exp_id}.sh")
    with open(script_path, "w") as f:
        f.write(f"""#!/usr/bin/env bash
set -e
cd {PROJ}
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES={gpu}
echo "[{exp_id} RESUME] from epoch_{resume_epoch}, total=60, gpu={gpu}"
pixi run sim train_dt_v2 \\
  train_dt_v2.dataset_dir={BASE}/{dataset} \\
  train_dt_v2.save_dir={model_dir} \\
  train_dt_v2.model.encoder_type={encoder} \\
  train_dt_v2.model.embed_dim=512 \\
  train_dt_v2.model.n_layer=6 \\
  train_dt_v2.model.n_head=8 \\
  train_dt_v2.optimizer.epochs=30 \\
  train_dt_v2.optimizer.total_epochs=60 \\
  train_dt_v2.optimizer.batch_size=512 \\
  train_dt_v2.optimizer.lr=1e-4 \\
  train_dt_v2.device=cuda \\
  train_dt_v2.resume_from={resume_ckpt} \\
  2>&1 | tee -a {LOG}/{exp_id}_{encoder}.log
echo "EXIT:$?"
""")
    os.chmod(script_path, 0o755)
    print(f"Written: resume_{exp_id}.sh (from epoch_{resume_epoch}, GPU {gpu})")

# E9/E10: 直接 60 epoch（无需 resume，首次训练）
E9E10 = [
    ("E9",  "dataset_D-D", "mlp",        0),
    ("E10", "dataset_D-D", "slice_attn", 1),
]
for exp_id, dataset, encoder, gpu in E9E10:
    model_dir = f"{BASE}/dt_model_{exp_id}_{encoder}"
    script_path = os.path.join(SCRIPTS, f"train_{exp_id}_60ep.sh")
    with open(script_path, "w") as f:
        f.write(f"""#!/usr/bin/env bash
set -e
cd {PROJ}
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES={gpu}
echo "[{exp_id}] dataset={dataset} encoder={encoder} gpu={gpu} epochs=60"
pixi run sim train_dt_v2 \\
  train_dt_v2.dataset_dir={BASE}/{dataset} \\
  train_dt_v2.save_dir={model_dir} \\
  train_dt_v2.model.encoder_type={encoder} \\
  train_dt_v2.model.embed_dim=512 \\
  train_dt_v2.model.n_layer=6 \\
  train_dt_v2.model.n_head=8 \\
  train_dt_v2.optimizer.epochs=60 \\
  train_dt_v2.optimizer.total_epochs=60 \\
  train_dt_v2.optimizer.batch_size=512 \\
  train_dt_v2.optimizer.lr=1e-4 \\
  train_dt_v2.device=cuda \\
  2>&1 | tee {LOG}/{exp_id}_{encoder}.log
echo "EXIT:$?"
""")
    os.chmod(script_path, 0o755)
    print(f"Written: train_{exp_id}_60ep.sh (60 epoch direct, GPU {gpu})")

print(f"\nAll resume/extension scripts in {SCRIPTS}/")
print("\n=== 启动建议 ===")
print("立即（E1-E4 空出 GPU2/GPU3 后）：")
print("  tmux new -s dt_E3r 'bash logs/dt_train_8exp/scripts/resume_E3.sh'")
print("  tmux new -s dt_E4r 'bash logs/dt_train_8exp/scripts/resume_E4.sh'")
print("E5/E6 完成后 GPU0/GPU1 空出：")
print("  tmux new -s dt_E1r 'bash logs/dt_train_8exp/scripts/resume_E1.sh'")
print("  tmux new -s dt_E2r 'bash logs/dt_train_8exp/scripts/resume_E2.sh'")
print("E9/E10 用 60-epoch 脚本（替换原 30ep 方案）：")
print("  在 E5/E6 完成前 kill dt_E1/dt_E2 的后续链，改用 60ep 脚本")
