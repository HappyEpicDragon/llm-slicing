"""
Oracle-Diverse Dataset Collection

Collect trajectories on TRAINING scenarios (0-4) with fixed (codebook, intra=RR)
actions. No PPO model needed — actions are directly constructed.

Outputs pkl files compatible with HierarchicalDTDataset, then builds violation-RTG
and computes metadata.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python scripts/collect_oracle_diverse.py
    .pixi/envs/default/bin/python scripts/collect_oracle_diverse.py --episodes 60 --parallel 5
"""
import os, sys, json, pickle, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_SLICES = 5
TRAIN_SCENARIOS = [0, 1, 2, 3, 4]
CODEBOOKS = list(range(11))
INTRA = 0  # RR

HP_WEIGHT = 10.0
NHP_WEIGHT = 1.0


def compute_step_viol_reward(obs_dict):
    """Per-step violation reward from observation (same as build_viol_rtg_dataset)."""
    inter = obs_dict["inter_feat"]
    r = 0.0
    for s in range(inter.shape[0]):
        drift = inter[s, 1]
        if drift < 0:
            w = HP_WEIGHT if inter[s, 0] > 0 else NHP_WEIGHT
            r += drift * w
    return r


def collect_scenario_codebooks(scenario_id, codebook_list, n_episodes, output_dir, mode):
    """Collect trajectories for one scenario with multiple codebooks."""
    import torch
    torch.set_num_threads(2)
    os.environ["OMP_NUM_THREADS"] = "2"

    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
    from src.basic_apis.network_slicing_business.path_manager import PathManager

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathManager(os.getcwd())

    save_dir = os.path.join(output_dir, mode)
    os.makedirs(save_dir, exist_ok=True)

    total_collected = 0

    for dim0 in codebook_list:
        action = np.array([[dim0] + [INTRA] * NUM_SLICES], dtype=np.int32)

        for seed in [10, 11]:  # 2 seeds for diversity
            cfg = env_cfg.env_settings.copy()
            cfg.mode = "training"
            cfg.scenario_mode = "inside"
            cfg.model_name = f"scenario_{scenario_id}"
            cfg.inside.training.active_scenario_list = [scenario_id]
            cfg.inside.training.max_scenario_episodes = (
                cfg.inside.training.init_scenario_episode + n_episodes
            )

            def _make(c=cfg, s=seed):
                def _init():
                    return HierarchicalSlicingEnv(c, np.random.default_rng(s), pm)
                return _init

            env = DummyVecEnv([_make()])
            obs = env.reset()

            curr_obs, curr_act, curr_rew, curr_viol_rew = [], [], [], []
            curr_done = []
            ep_count = 0

            try:
                while ep_count < n_episodes:
                    step_obs = {k: v[0].copy() for k, v in obs.items()}
                    curr_obs.append(step_obs)
                    curr_act.append(action[0].copy())

                    next_obs, rewards, dones, infos = env.step(action)
                    curr_rew.append(rewards[0])
                    curr_done.append(dones[0])

                    viol_r = compute_step_viol_reward(step_obs)
                    curr_viol_rew.append(viol_r)

                    obs = next_obs

                    if dones[0]:
                        viol_rewards = np.array(curr_viol_rew, dtype=np.float64)
                        traj_data = {
                            "observations": curr_obs,
                            "actions": np.array(curr_act, dtype=np.float32),
                            "rewards": viol_rewards.astype(np.float32),
                            "dones": np.array(curr_done, dtype=bool),
                            "episode_returns": float(viol_rewards.sum()),
                            "episode_length": len(curr_act),
                            "hp_viols": 0,
                            "nhp_viols": 0,
                            "meta_agent": dim0,
                            "meta_scen": scenario_id,
                            "meta_phys_ep": ep_count,
                        }

                        ret_str = f"{traj_data['episode_returns']:.0f}"
                        pkl_name = f"S{scenario_id}_D{dim0}_sd{seed}_ep{ep_count}_ret{ret_str}.pkl"
                        with open(os.path.join(save_dir, pkl_name), "wb") as f:
                            pickle.dump(traj_data, f)

                        total_collected += 1
                        ep_count += 1
                        curr_obs, curr_act, curr_rew, curr_viol_rew, curr_done = [], [], [], [], []
            finally:
                env.close()

    return scenario_id, total_collected


def compute_metadata(output_dir):
    """Compute obs normalization stats and action distribution."""
    train_dir = os.path.join(output_dir, "training")
    if not os.path.exists(train_dir):
        return

    stats = {
        "inter": {"sum": 0, "sq_sum": 0, "count": 0},
        "intra": {"sum": 0, "sq_sum": 0, "count": 0},
        "global": {"sum": 0, "sq_sum": 0, "count": 0}
    }
    scenario_returns = {}
    action_counts = {}
    source_stats = {}

    files = sorted([f for f in os.listdir(train_dir) if f.endswith('.pkl')])
    print(f"Computing metadata from {len(files)} trajectories...")

    for f_name in tqdm(files, desc="Metadata"):
        try:
            with open(os.path.join(train_dir, f_name), "rb") as f:
                traj = pickle.load(f)
        except:
            continue

        scen_id = traj.get('meta_scen', -1)
        agent_id = traj.get('meta_agent', -1)
        ep_ret = traj["episode_returns"]

        if scen_id not in scenario_returns:
            scenario_returns[scen_id] = []
        scenario_returns[scen_id].append(ep_ret)

        if scen_id not in source_stats:
            source_stats[scen_id] = {}
        source_stats[scen_id][agent_id] = source_stats[scen_id].get(agent_id, 0) + 1

        actions = traj['actions']
        if actions.ndim == 2:
            T, dims = actions.shape
            for d in range(dims):
                if d not in action_counts:
                    action_counts[d] = {}
                vals, counts = np.unique(actions[:, d].astype(int), return_counts=True)
                for v, c in zip(vals, counts):
                    v = int(v)
                    action_counts[d][v] = action_counts[d].get(v, 0) + int(c)

        inter = np.array([o['inter_feat'] for o in traj['observations']])
        intra = np.array([o['intra_feat'] for o in traj['observations']])
        glob = np.array([o['global_feat'] for o in traj['observations']])

        flat_inter = inter.reshape(-1, 4)
        stats["inter"]["sum"] += flat_inter.sum(axis=0)
        stats["inter"]["sq_sum"] += (flat_inter ** 2).sum(axis=0)
        stats["inter"]["count"] += flat_inter.shape[0]

        flat_intra = intra.reshape(-1, 5)
        stats["intra"]["sum"] += flat_intra.sum(axis=0)
        stats["intra"]["sq_sum"] += (flat_intra ** 2).sum(axis=0)
        stats["intra"]["count"] += flat_intra.shape[0]

        flat_glob = glob.reshape(-1, 2)
        stats["global"]["sum"] += flat_glob.sum(axis=0)
        stats["global"]["sq_sum"] += (flat_glob ** 2).sum(axis=0)
        stats["global"]["count"] += flat_glob.shape[0]

    metadata = {
        "obs_stats": {},
        "scenario_stats": {},
        "action_dist": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in action_counts.items()},
        "expert_contrib": {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in source_stats.items()},
    }

    for k in stats:
        N = stats[k]["count"]
        if N == 0:
            continue
        mean = stats[k]["sum"] / N
        var = (stats[k]["sq_sum"] / N) - (mean ** 2)
        std = np.sqrt(np.maximum(var, 1e-6))
        metadata["obs_stats"][k] = {"mean": mean.tolist(), "std": std.tolist()}

    print("\nScenario Return Stats (violation-based):")
    for s_id in sorted(scenario_returns.keys()):
        rets = np.array(scenario_returns[s_id])
        meta = {
            "min": float(np.min(rets)), "mean": float(np.mean(rets)),
            "max": float(np.max(rets)), "p95": float(np.percentile(rets, 95))
        }
        metadata["scenario_stats"][str(s_id)] = meta
        print(f"  S{s_id}: mean={meta['mean']:.0f}, range=[{meta['min']:.0f}, {meta['max']:.0f}]")

    print("\nAction Distribution (dim0):")
    if "0" in metadata["action_dist"]:
        d0_dist = metadata["action_dist"]["0"]
        total_steps = sum(d0_dist.values())
        for act_val in sorted(d0_dist.keys(), key=int):
            pct = d0_dist[act_val] / total_steps * 100
            print(f"  act{act_val}: {d0_dist[act_val]:>8} ({pct:.1f}%)")

    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nMetadata saved to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Oracle-Diverse dataset collection")
    parser.add_argument("--episodes", type=int, default=40,
                        help="Episodes per (scenario, codebook, seed)")
    parser.add_argument("--parallel", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="data/channel_generality/dt_oracle_diverse/dataset")
    args = parser.parse_args()

    n_configs = len(TRAIN_SCENARIOS) * len(CODEBOOKS) * 2  # 2 seeds
    total_eps = n_configs * args.episodes
    total_traj = len(TRAIN_SCENARIOS) * len(CODEBOOKS) * 2 * args.episodes
    print(f"Oracle-Diverse Collection")
    print(f"  {len(TRAIN_SCENARIOS)} scenarios × {len(CODEBOOKS)} codebooks × 2 seeds × {args.episodes} eps")
    print(f"  = {total_traj} total trajectories ({total_traj * 1000 / 1e6:.1f}M steps)")
    print(f"  Output: {args.output}")
    print(f"  Parallel: {args.parallel}")
    print()

    os.makedirs(os.path.join(args.output, "training"), exist_ok=True)

    t0 = time.time()

    if args.parallel > 1:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {}
            for scen in TRAIN_SCENARIOS:
                fut = pool.submit(
                    collect_scenario_codebooks, scen, CODEBOOKS,
                    args.episodes, args.output, "training"
                )
                futures[fut] = scen

            for fut in as_completed(futures):
                scen = futures[fut]
                try:
                    sid, cnt = fut.result()
                    elapsed = time.time() - t0
                    print(f"  [{elapsed/60:.1f}min] S{sid}: {cnt} trajectories collected")
                except Exception as e:
                    print(f"  FAILED S{scen}: {e}")
                    import traceback; traceback.print_exc()
    else:
        for scen in TRAIN_SCENARIOS:
            print(f"\n--- Scenario {scen} ---")
            sid, cnt = collect_scenario_codebooks(
                scen, CODEBOOKS, args.episodes, args.output, "training"
            )
            elapsed = time.time() - t0
            print(f"  [{elapsed/60:.1f}min] {cnt} trajectories")

    total_time = time.time() - t0
    print(f"\nCollection done in {total_time/60:.1f} min")

    print("\n" + "="*60)
    print("  Computing metadata...")
    print("="*60)
    compute_metadata(args.output)

    n_files = len([f for f in os.listdir(os.path.join(args.output, "training")) if f.endswith('.pkl')])
    print(f"\nDataset ready: {args.output}")
    print(f"  {n_files} training trajectories")
    print(f"\nTo train DT:")
    print(f"  pixi run sim dt_training \\")
    print(f"    dt_training.train_dt.dataset.base_dir={os.path.abspath(args.output)}/ \\")
    print(f"    dt_training.train_dt.dataset.meta_path={os.path.abspath(args.output)}/metadata.json \\")
    print(f"    dt_training.train_dt.evaluation.target_rtg=0.0 \\")
    print(f"    dt_training.train_dt.saving.asset.model_root=data/channel_generality/dt_oracle_diverse/model")


if __name__ == "__main__":
    main()
