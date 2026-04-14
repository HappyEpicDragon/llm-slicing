"""
Collect trajectories from multiple per-scenario PPO experts in parallel.
Each scenario uses its own expert model.

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/collect_multi_expert.py \
        --expert_root data/channel_generality/dt_v2_per_scenario \
        --scenarios 0 1 2 3 4 \
        --output data/channel_generality/dt_v2_expert_s0to4/dataset \
        --episodes 200 --epsilon 0.1 --workers 5
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.collect_data_v2 import collect_scenario, compute_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert_root",
        type=str,
        default="data/channel_generality/dt_v2_per_scenario",
        help="Root dir containing ppo_s{sid}/best_model/best_model.zip",
    )
    parser.add_argument(
        "--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4],
        help="Scenarios to collect from",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/channel_generality/dt_v2_expert_s0to4/dataset",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--teacher_kind",
        type=str,
        default="per_scenario",
        help="Teacher source tag written into each trajectory",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Parallel processes (one per scenario is fine)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output, "training"), exist_ok=True)

    # Build (scenario, model_path) pairs
    tasks = []
    for scen in args.scenarios:
        model_path = os.path.join(
            ROOT, args.expert_root, f"ppo_s{scen}", "best_model", "best_model.zip"
        )
        if not os.path.isfile(model_path):
            print(f"[WARN] S{scen}: model not found at {model_path}, skipping", flush=True)
            continue
        teacher_id = f"ppo_s{scen}"
        tasks.append((scen, model_path, teacher_id))

    total = len(tasks) * args.episodes
    print(
        f"Multi-expert collection: {len(tasks)} scenarios × {args.episodes} eps "
        f"= {total} trajectories",
        flush=True,
    )
    for scen, mp, teacher_id in tasks:
        print(f"  S{scen}: {mp} ({args.teacher_kind}/{teacher_id})", flush=True)
    print(f"  Output: {args.output}\n", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for scen, model_path, teacher_id in tasks:
            fut = pool.submit(
                collect_scenario,
                scen, model_path, args.episodes,
                args.output, args.epsilon, args.seed,
                args.teacher_kind, teacher_id,
            )
            futures[fut] = scen

        for fut in as_completed(futures):
            scen = futures[fut]
            try:
                sid, cnt = fut.result()
                elapsed = (time.time() - t0) / 60
                print(f"  [{elapsed:.1f}min] S{sid}: {cnt} trajectories done", flush=True)
            except Exception as e:
                import traceback
                print(f"  [ERROR] S{scen}: {e}", flush=True)
                traceback.print_exc()

    print(f"\nAll collection done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Compute metadata once over ALL collected trajectories
    print("\nComputing metadata over all collected data...", flush=True)
    compute_metadata(args.output)

    n_files = len(
        [f for f in os.listdir(os.path.join(args.output, "training")) if f.endswith(".pkl")]
    )
    print(f"\nDataset ready: {args.output}/training/ ({n_files} files)", flush=True)


if __name__ == "__main__":
    main()
