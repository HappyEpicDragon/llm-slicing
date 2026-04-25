"""
Phase 3a: Collect PPO teacher trajectories on EnvV2 with epsilon-greedy.

Saves pkl files compatible with HierarchicalDTDataset (same format as
collect_oracle_diverse.py but with PPO-generated actions).

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.collect_data_v2
"""
import os, sys, json, pickle, time, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

NUM_SLICES = 5
HP_WEIGHT = 10.0
NHP_WEIGHT = 1.0


def compute_step_viol_reward(inter_feat):
    """Per-step violation reward from inter_feat (v2: 8 dims)."""
    r = 0.0
    for s in range(inter_feat.shape[0]):
        drift = inter_feat[s, 1]
        prio = inter_feat[s, 0]
        if drift < 0:
            w = HP_WEIGHT if prio > 0 else NHP_WEIGHT
            r += drift * w
    return r


def _safe_tag(raw):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))


def infer_teacher_id(model_path):
    path = Path(model_path)
    if len(path.parents) >= 2:
        return path.parents[1].name
    return path.stem


def collect_scenario(
    scenario_id,
    model_path,
    n_episodes,
    output_dir,
    epsilon,
    seed,
    teacher_kind="single",
    teacher_id=None,
):
    """Collect trajectories for one scenario using PPO teacher."""
    import torch
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"

    from omegaconf import OmegaConf
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
    from src.basic_apis.network_slicing_business.path_context import PathContext

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathContext(os.getcwd())

    cfg = env_cfg.env_settings.copy()
    cfg.mode = "training"
    cfg.scenario_mode = "inside"
    cfg.model_name = f"scenario_{scenario_id}"
    cfg.inside.training.active_scenario_list = [scenario_id]
    # Do NOT expand max_scenario_episodes. Keep the training-mode cap (ep 0–59)
    # so the environment wraps within training episodes when n_episodes > 60.
    # Expanding the cap would cause the env to access validation/test episodes
    # (ep 60–99), which would be data leakage for in-distribution evaluation.

    def _make(c=cfg, s=seed):
        def _init():
            return HierarchicalSlicingEnvV2(c, np.random.default_rng(s), pm)
        return _init

    env = DummyVecEnv([_make()])
    model = PPO.load(model_path, device="cpu")

    rng = np.random.default_rng(seed + scenario_id * 1000)
    teacher_id = teacher_id or infer_teacher_id(model_path)
    save_dir = os.path.join(output_dir, "training")
    os.makedirs(save_dir, exist_ok=True)

    obs = env.reset()
    collected = 0
    curr_obs, curr_act, curr_viol_rew, curr_done = [], [], [], []

    try:
        while collected < n_episodes:
            if rng.random() < epsilon:
                action = env.action_space.sample().reshape(1, -1)
            else:
                action, _ = model.predict(obs, deterministic=False)
                action = action.reshape(1, -1)

            step_obs = {k: v[0].copy() for k, v in obs.items()}
            curr_obs.append(step_obs)
            curr_act.append(action[0].copy())

            viol_r = compute_step_viol_reward(step_obs["inter_feat"])
            curr_viol_rew.append(viol_r)

            next_obs, rewards, dones, infos = env.step(action)
            curr_done.append(dones[0])
            obs = next_obs

            if dones[0]:
                viol_rewards = np.array(curr_viol_rew, dtype=np.float32)
                traj = {
                    "observations": curr_obs,
                    "actions": np.array(curr_act, dtype=np.float32),
                    "rewards": viol_rewards,
                    "dones": np.array(curr_done, dtype=bool),
                    "episode_returns": float(viol_rewards.sum()),
                    "episode_length": len(curr_act),
                    "hp_viols": 0,
                    "nhp_viols": 0,
                    "meta_agent": 0,
                    "meta_scen": scenario_id,
                    "meta_phys_ep": collected,
                    "meta_teacher_kind": teacher_kind,
                    "meta_teacher_id": teacher_id,
                }

                ret_str = f"{traj['episode_returns']:.0f}"
                fname = (
                    f"S{scenario_id}_{_safe_tag(teacher_kind)}_{_safe_tag(teacher_id)}"
                    f"_ep{collected}_ret{ret_str}.pkl"
                )
                with open(os.path.join(save_dir, fname), "wb") as f:
                    pickle.dump(traj, f)

                collected += 1
                curr_obs, curr_act, curr_viol_rew, curr_done = [], [], [], []
    finally:
        env.close()

    return scenario_id, collected


def compute_metadata(output_dir):
    """Compute obs normalization stats (same logic as collect_oracle_diverse)."""
    train_dir = os.path.join(output_dir, "training")
    files = sorted([f for f in os.listdir(train_dir) if f.endswith('.pkl')])
    print(f"Computing metadata from {len(files)} trajectories...")

    inter_dim = 8
    intra_dim = 7
    global_dim = 2

    stats = {
        "inter": {"sum": np.zeros(inter_dim), "sq_sum": np.zeros(inter_dim), "count": 0},
        "intra": {"sum": np.zeros(intra_dim), "sq_sum": np.zeros(intra_dim), "count": 0},
        "global": {"sum": np.zeros(global_dim), "sq_sum": np.zeros(global_dim), "count": 0},
    }
    scenario_returns = defaultdict(list)
    action_counts = {}
    teacher_source_counts = defaultdict(int)
    teacher_id_counts = defaultdict(int)
    teacher_contrib = defaultdict(lambda: defaultdict(int))

    for f_name in tqdm(files, desc="Metadata"):
        with open(os.path.join(train_dir, f_name), "rb") as f:
            traj = pickle.load(f)

        scen_id = traj.get("meta_scen", -1)
        teacher_kind = traj.get("meta_teacher_kind", "unknown")
        teacher_id = traj.get("meta_teacher_id", "unknown")
        scenario_returns[scen_id].append(traj["episode_returns"])
        teacher_source_counts[teacher_kind] += 1
        teacher_id_counts[teacher_id] += 1
        teacher_contrib[scen_id][teacher_id] += 1

        actions = traj["actions"]
        if actions.ndim == 2:
            for d in range(actions.shape[1]):
                if d not in action_counts:
                    action_counts[d] = {}
                vals, counts = np.unique(actions[:, d].astype(int), return_counts=True)
                for v, c in zip(vals, counts):
                    action_counts[d][int(v)] = action_counts[d].get(int(v), 0) + int(c)

        for obs in traj["observations"]:
            for key, dim in [("inter_feat", inter_dim), ("intra_feat", intra_dim), ("global_feat", global_dim)]:
                arr = obs[key]
                flat = arr.reshape(-1, dim)
                stat_key = key.split("_")[0]
                stats[stat_key]["sum"] += flat.sum(axis=0)
                stats[stat_key]["sq_sum"] += (flat ** 2).sum(axis=0)
                stats[stat_key]["count"] += flat.shape[0]

    metadata = {
        "obs_stats": {},
        "scenario_stats": {},
        "action_dist": {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in action_counts.items()
        },
        "teacher_source_counts": dict(sorted(teacher_source_counts.items())),
        "teacher_id_counts": dict(sorted(teacher_id_counts.items())),
        "teacher_contrib": {
            str(s_id): dict(sorted(src_counts.items()))
            for s_id, src_counts in sorted(teacher_contrib.items())
        },
    }

    for k in stats:
        N = stats[k]["count"]
        if N == 0:
            continue
        mean = stats[k]["sum"] / N
        var = (stats[k]["sq_sum"] / N) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-6))
        metadata["obs_stats"][k] = {"mean": mean.tolist(), "std": std.tolist()}

    for s_id in sorted(scenario_returns.keys()):
        rets = np.array(scenario_returns[s_id])
        metadata["scenario_stats"][str(s_id)] = {
            "min": float(np.min(rets)), "mean": float(np.mean(rets)),
            "max": float(np.max(rets)), "p95": float(np.percentile(rets, 95)),
        }

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    print("\nScenario return stats:")
    for s_id in sorted(scenario_returns.keys()):
        rets = np.array(scenario_returns[s_id])
        print(f"  S{s_id}: mean={rets.mean():.0f}, range=[{rets.min():.0f}, {rets.max():.0f}]")

    print(f"\nAction distribution (dim0 = codebook):")
    d0 = action_counts.get(0, {})
    total = sum(d0.values()) if d0 else 0
    for a in sorted(d0.keys()):
        print(f"  act{a}: {d0[a]} ({d0[a]/total*100:.1f}%)")

    print(f"\nAction distribution (dim1 = ordering_slice0):")
    d1 = action_counts.get(1, {})
    total1 = sum(d1.values()) if d1 else 0
    for a in sorted(d1.keys()):
        print(f"  priority{a}: {d1[a]} ({d1[a]/total1*100:.1f}%)")

    print("\nTeacher source counts:")
    for teacher_kind, cnt in sorted(teacher_source_counts.items()):
        print(f"  {teacher_kind}: {cnt}")

    print("\nTeacher ID counts:")
    for teacher_id, cnt in sorted(teacher_id_counts.items()):
        print(f"  {teacher_id}: {cnt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="data/channel_generality/dt_v2/ppo_teacher/best_model/best_model.zip")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=str, default="data/channel_generality/dt_v2/dataset")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--teacher_kind", type=str, default="single")
    parser.add_argument("--teacher_id", type=str, default=None)
    args = parser.parse_args()

    total = len(args.scenarios) * args.episodes
    print(f"Collecting PPO teacher trajectories (epsilon={args.epsilon})")
    print(f"  {len(args.scenarios)} scenarios × {args.episodes} episodes = {total} trajectories")
    print(f"  Model: {args.model}")
    print(f"  Teacher kind: {args.teacher_kind}")
    if args.teacher_id:
        print(f"  Teacher id: {args.teacher_id}")
    print(f"  Output: {args.output}\n")

    os.makedirs(os.path.join(args.output, "training"), exist_ok=True)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {}
        for scen in args.scenarios:
            fut = pool.submit(
                collect_scenario,
                scen,
                args.model,
                args.episodes,
                args.output,
                args.epsilon,
                args.seed,
                args.teacher_kind,
                args.teacher_id,
            )
            futures[fut] = scen

        for fut in as_completed(futures):
            scen = futures[fut]
            try:
                sid, cnt = fut.result()
                elapsed = time.time() - t0
                print(f"  [{elapsed/60:.1f}min] S{sid}: {cnt} trajectories")
            except Exception as e:
                print(f"  FAILED S{scen}: {e}")
                import traceback; traceback.print_exc()

    print(f"\nCollection done in {(time.time()-t0)/60:.1f} min")
    print("\nComputing metadata...")
    compute_metadata(args.output)

    n_files = len([f for f in os.listdir(os.path.join(args.output, "training")) if f.endswith(".pkl")])
    print(f"\nDataset ready: {args.output}/training/ ({n_files} files)")


def collect(cfg, path_context):
    """Hydra entry point: called by channel_generality.py → collect_data_v2 mode."""
    tc = cfg.collect_data_v2
    model_path = str(tc.model_path)
    output = str(tc.output)
    scenarios = list(tc.scenarios)
    n_episodes = int(tc.n_episodes)
    epsilon = float(tc.epsilon)
    seed = int(tc.seed)
    parallel = int(tc.parallel)
    teacher_kind = str(tc.teacher_kind)
    teacher_id = str(tc.teacher_id) if tc.get("teacher_id") else None

    print(f"[collect_data_v2] {len(scenarios)} scenarios × {n_episodes} episodes")
    print(f"  model: {model_path}, teacher_kind: {teacher_kind}")
    print(f"  output: {output}")

    os.makedirs(os.path.join(output, "training"), exist_ok=True)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {}
        for scen in scenarios:
            fut = pool.submit(
                collect_scenario,
                scen, model_path, n_episodes, output,
                epsilon, seed, teacher_kind, teacher_id,
            )
            futures[fut] = scen

        for fut in as_completed(futures):
            scen = futures[fut]
            try:
                sid, cnt = fut.result()
                print(f"  [{(time.time()-t0)/60:.1f}min] S{sid}: {cnt} trajectories")
            except Exception as e:
                print(f"  FAILED S{scen}: {e}")
                import traceback; traceback.print_exc()

    print(f"\nCollection done in {(time.time()-t0)/60:.1f} min")
    compute_metadata(output)
    n_files = len([f for f in os.listdir(os.path.join(output, "training")) if f.endswith(".pkl")])
    print(f"Dataset ready: {output}/training/ ({n_files} files)")


if __name__ == "__main__":
    main()
