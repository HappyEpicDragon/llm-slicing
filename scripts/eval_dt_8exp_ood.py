"""
评估 10 个 DT 实验模型在 s5-s9 上的 OOD 性能，并与 ppo-multi 比较。

每个模型使用其当前最新的 checkpoint（epoch_60 > epoch_30 > ...）。
结果保存到 data/channel_generality/dt_v2_8exp/eval_ood/
最后打印对比表。
"""
import os, json, argparse, subprocess, sys
from pathlib import Path

PROJ = Path("/root/decision_transformer_slicing")
BASE = PROJ / "data/channel_generality/dt_v2_8exp"
EVAL_ROOT = BASE / "eval_ood"
LOG_DIR = PROJ / "logs/dt_eval_8exp"
LOG_DIR.mkdir(parents=True, exist_ok=True)
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

META_PATH = str(BASE / "dataset_D-A/metadata.json")

EXPERIMENTS = [
    # (exp_id, dataset, encoder, gpu)
    ("E1",  "dataset_D-A", "mlp",        2),
    ("E2",  "dataset_D-A", "slice_attn", 2),
    ("E3",  "dataset_D-B", "mlp",        2),
    ("E4",  "dataset_D-B", "slice_attn", 2),
    ("E5",  "dataset_D-C", "mlp",        2),
    ("E6",  "dataset_D-C", "slice_attn", 3),
    ("E7",  "dataset_D-E", "mlp",        3),
    ("E8",  "dataset_D-E", "slice_attn", 3),
    ("E9",  "dataset_D-D", "mlp",        3),
    ("E10", "dataset_D-D", "slice_attn", 3),
]

def latest_checkpoint(model_dir: Path) -> Path | None:
    """返回最新 epoch_N.pth（按 N 数值排序）。"""
    ckpts = sorted(
        model_dir.glob("epoch_*.pth"),
        key=lambda p: int(p.stem.split("_")[1])
    )
    return ckpts[-1] if ckpts else None


def gen_eval_script(exp_id, encoder, model_path, save_root, gpu, log_path):
    script = f"""#!/usr/bin/env bash
set -e
cd {PROJ}
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES={gpu}
echo "[{exp_id}] checkpoint={model_path}"
pixi run sim test_dt_v2 \\
  test_dt_v2.model_path={model_path} \\
  test_dt_v2.meta_path={META_PATH} \\
  test_dt_v2.encoder_type={encoder} \\
  test_dt_v2.model.n_layer=6 \\
  test_dt_v2.model.n_head=8 \\
  test_dt_v2.model.embed_dim=512 \\
  test_dt_v2.test_scenarios=[5,6,7,8,9] \\
  test_dt_v2.test_seeds=[0] \\
  test_dt_v2.n_episodes=20 \\
  test_dt_v2.device=cuda \\
  test_dt_v2.save_root={save_root} \\
  2>&1 | tee {log_path}
echo "EXIT:$?"
"""
    return script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="只生成脚本，不启动")
    args = parser.parse_args()

    scripts_dir = LOG_DIR / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    print("=== DT 8-Exp OOD Evaluation ===")
    print(f"{'Exp':<6} {'Encoder':<12} {'Latest Ckpt':<10} {'Path'}")
    print("-" * 80)

    script_paths = {}
    for exp_id, dataset, encoder, gpu in EXPERIMENTS:
        model_dir = BASE / f"dt_model_{exp_id}_{encoder}"
        ckpt = latest_checkpoint(model_dir)
        if ckpt is None:
            print(f"{exp_id:<6} {encoder:<12} {'MISSING':<10} {model_dir}")
            continue
        epoch_num = int(ckpt.stem.split("_")[1])
        print(f"{exp_id:<6} {encoder:<12} {'ep'+str(epoch_num):<10} {ckpt}")

        save_root = EVAL_ROOT / exp_id
        log_path = LOG_DIR / f"{exp_id}_{encoder}_ood.log"
        script_content = gen_eval_script(
            exp_id, encoder, str(ckpt), str(save_root), gpu, str(log_path)
        )
        script_path = scripts_dir / f"eval_{exp_id}_ood.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        script_paths[exp_id] = (script_path, encoder, gpu)

    print()
    if args.dry_run:
        print("Dry run: scripts generated only.")
        for exp_id, (sp, enc, gpu) in script_paths.items():
            print(f"  {sp}")
        return

    # GPU2 上的模型（E1-E5）串行运行
    gpu2_exps = [(eid, sp) for eid, (sp, enc, gpu) in script_paths.items() if gpu == 2]
    gpu3_exps = [(eid, sp) for eid, (sp, enc, gpu) in script_paths.items() if gpu == 3]

    chain2 = " && ".join([f"bash {sp}" for _, sp in gpu2_exps])
    chain3 = " && ".join([f"bash {sp}" for _, sp in gpu3_exps])

    print("启动 tmux 评估会话...")
    # 写链式 wrapper 脚本再 tmux 执行
    for gpu_id, exps, session in [(2, gpu2_exps, "dt_eval_gpu2"), (3, gpu3_exps, "dt_eval_gpu3")]:
        if not exps:
            continue
        chain_script = scripts_dir / f"chain_gpu{gpu_id}.sh"
        with open(chain_script, "w") as f:
            f.write("#!/usr/bin/env bash\nset -e\n")
            for eid, sp in exps:
                f.write(f"echo '=== {eid} ==='\nbash {sp}\n")
        chain_script.chmod(0o755)
        os.system(f"tmux new-session -d -s {session} 'bash {chain_script}'")
        print(f"  {session}: {[e for e,_ in exps]}")

    print()
    print("日志目录:", LOG_DIR)
    print("结果目录:", EVAL_ROOT)
    print()
    print("监控命令:")
    print("  tmux attach -t dt_eval_gpu2")
    print("  tmux attach -t dt_eval_gpu3")
    print("  tail -f logs/dt_eval_8exp/E3_mlp_ood.log")


if __name__ == "__main__":
    main()
