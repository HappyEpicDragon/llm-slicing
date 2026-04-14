"""
生成 S3 穷举实验的批量训练+评测 shell 脚本。
跳过已有 eval summary 的子集。每 GPU 串行执行分配给它的任务。
"""
import os
from itertools import combinations
from pathlib import Path

BASE = Path("data/channel_generality/sensitivity/s3_exhaustive")
EXISTING_EVAL = Path("data/channel_generality/sensitivity/s3_coverage")
E4_DIR = Path("data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100")
LOG_DIR = Path("data/channel_generality/sensitivity/logs/exhaustive")
SCRIPT_DIR = Path("scripts")
NUM_GPUS = 4

ALL_SCENARIOS = [0, 1, 2, 3, 4]
TEST_SCENARIOS = [5, 6, 7, 8, 9]


def subset_key(ids):
    return "_".join(str(s) for s in sorted(ids))


def has_eval(key):
    d = BASE / key / "eval"
    return all(
        (d / f"scenario_{s}" / "summary.json").exists()
        for s in TEST_SCENARIOS
    )


def check_old_eval(scen_ids):
    """Check if results exist in old s3_coverage directories."""
    k = len(scen_ids)
    ids = sorted(scen_ids)
    if k == 5:
        return all((E4_DIR / f"scenario_{s}" / "summary.json").exists() for s in TEST_SCENARIOS)
    if ids == [0] and (EXISTING_EVAL / "scen1" / "eval" / "scenario_5" / "summary.json").exists():
        return True
    if ids == [0, 1] and (EXISTING_EVAL / "scen2" / "eval" / "scenario_5" / "summary.json").exists():
        return True
    if ids == [0, 1, 2] and (EXISTING_EVAL / "scen3" / "eval" / "scenario_5" / "summary.json").exists():
        return True
    if ids == [0, 1, 2, 3] and (EXISTING_EVAL / "scen4" / "eval" / "scenario_5" / "summary.json").exists():
        return True
    if ids == [0, 1, 2, 4] and (EXISTING_EVAL / "scen4_v2" / "eval" / "scenario_5" / "summary.json").exists():
        return True
    return False


def gen_train_eval_block(key, gpu_id):
    ds_dir = f"data/channel_generality/sensitivity/s3_exhaustive/{key}"
    log_file = f"data/channel_generality/sensitivity/logs/exhaustive/{key}.log"
    return f"""
echo "=== {key} gpu={gpu_id} train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES={gpu_id} /root/.pixi/bin/pixi run sim train_dt_v2_tiny \\
    train_dt_v2_tiny.grid_id=b2_e2 \\
    train_dt_v2_tiny.dataset_dir={ds_dir} \\
    train_dt_v2_tiny.save_dir={ds_dir}/model \\
    train_dt_v2_tiny.model.context_len=20 \\
    train_dt_v2_tiny.model.embed_dim=32 \\
    train_dt_v2_tiny.model.n_layer=2 \\
    train_dt_v2_tiny.model.n_head=2 \\
    train_dt_v2_tiny.model.encoder_type=slice_attn \\
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \\
    train_dt_v2_tiny.model.encoder_num_heads=2 \\
    train_dt_v2_tiny.optimizer.epochs=30 \\
    train_dt_v2_tiny.optimizer.batch_size=1024 \\
    train_dt_v2_tiny.optimizer.lr=1e-4 \\
    train_dt_v2_tiny.device=cuda > {log_file} 2>&1

echo "=== {key} gpu={gpu_id} eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES={gpu_id} /root/.pixi/bin/pixi run sim test_dt_v2_tiny \\
    test_dt_v2_tiny.grid_id=b2_e2 \\
    test_dt_v2_tiny.model_path={ds_dir}/model/final_dt_v2.pth \\
    test_dt_v2_tiny.meta_path={ds_dir}/metadata.json \\
    test_dt_v2_tiny.model.context_len=20 \\
    test_dt_v2_tiny.model.embed_dim=32 \\
    test_dt_v2_tiny.model.n_layer=2 \\
    test_dt_v2_tiny.model.n_head=2 \\
    test_dt_v2_tiny.encoder_type=slice_attn \\
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \\
    test_dt_v2_tiny.model.encoder_num_heads=2 \\
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \\
    'test_dt_v2_tiny.test_seeds=[0]' \\
    test_dt_v2_tiny.n_episodes=20 \\
    test_dt_v2_tiny.init_episode=0 \\
    test_dt_v2_tiny.max_episode=100 \\
    test_dt_v2_tiny.target_rtg=0 \\
    test_dt_v2_tiny.save_root={ds_dir}/eval \\
    test_dt_v2_tiny.device=cuda >> {log_file} 2>&1

echo "=== {key} DONE $(date) ==="
"""


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    jobs = []
    for k in range(1, 5):
        for combo in combinations(ALL_SCENARIOS, k):
            ids = sorted(combo)
            key = f"k{k}_{subset_key(ids)}"
            if has_eval(key) or check_old_eval(ids):
                print(f"  [SKIP] {key}")
                continue
            jobs.append(key)

    print(f"\nTotal jobs to run: {len(jobs)}")
    for i, j in enumerate(jobs):
        print(f"  [{i}] {j}")

    gpu_queues = [[] for _ in range(NUM_GPUS)]
    for i, job in enumerate(jobs):
        gpu_queues[i % NUM_GPUS].append(job)

    for gpu_id in range(NUM_GPUS):
        if not gpu_queues[gpu_id]:
            continue
        script_path = SCRIPT_DIR / f"s3_exhaust_gpu{gpu_id}.sh"
        with open(script_path, "w") as f:
            f.write("#!/bin/bash\nset -euo pipefail\n")
            f.write("cd /root/decision_transformer_slicing\n\n")
            for key in gpu_queues[gpu_id]:
                f.write(gen_train_eval_block(key, gpu_id))
                f.write("\n")
            f.write(f'\necho "=== GPU {gpu_id} ALL JOBS DONE $(date) ==="\n')
        os.chmod(script_path, 0o755)
        print(f"  Written {script_path} ({len(gpu_queues[gpu_id])} jobs)")

    print("\nLaunch with:")
    for gpu_id in range(NUM_GPUS):
        if gpu_queues[gpu_id]:
            print(f"  tmux new-session -d -s s3_gpu{gpu_id} 'bash scripts/s3_exhaust_gpu{gpu_id}.sh'")
