"""
Phase 3c: Test DT on EnvV2 (test scenarios 5-9).

Runs DT model inference with the new 11-dim action space on EnvV2,
measuring violation metrics for comparison with PPO and old DT.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python -m src.basic_apis.dt_v2.test_v2 --model_path <path>
"""
import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2, INTER_DIM, INTRA_DIM
from src.basic_apis.dt_v2.model_v2 import build_dt_v2
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.general_utils import pad_stack_tensor

NUM_SLICES = 5


def _load_obs_stats(meta_path):
    if not meta_path or not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    obs_stats_raw = meta.get("obs_stats", meta)
    stats = {}
    for k in ["inter", "intra", "global"]:
        if k in obs_stats_raw:
            stats[k] = {
                "mean": np.array(obs_stats_raw[k]["mean"], dtype=np.float32),
                "std": np.array(obs_stats_raw[k]["std"], dtype=np.float32) + 1e-6,
            }
    return stats if stats else None


def _normalize_obs(raw, feat_key, obs_stats):
    if obs_stats is None:
        return raw
    stat_key = feat_key.split("_")[0]
    if stat_key not in obs_stats:
        return raw
    mean = obs_stats[stat_key]["mean"]
    std = obs_stats[stat_key]["std"]
    safe_std = std.copy()
    safe_std[safe_std < 1e-2] = 1.0
    normed = (raw - mean) / safe_std
    return np.clip(normed, -5.0, 5.0).astype(np.float32)


def _compute_step_metrics(info, num_slices=NUM_SLICES):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        mean_drift = float(np.mean(drifts))
        if mean_drift < 0:
            if is_hp:
                hp_viols += 1
                hp_dist += mean_drift
            else:
                nhp_viols += 1
                nhp_dist += mean_drift
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def run_dt_episode(env, model, device, target_rtg, context_len, action_dims, obs_stats):
    obs = env.reset()
    dummy_action = torch.zeros(len(action_dims), dtype=torch.long, device=device)
    history = {"inter": [], "intra": [], "global": [], "actions": [dummy_action], "rtg": [], "timesteps": []}

    cumulative_reward = 0.0
    hp_viol_sum = nhp_viol_sum = 0
    hp_dist_sum = nhp_dist_sum = 0.0
    hp_active_t = nhp_active_t = 0
    step_hp_dists, step_nhp_dists, step_rewards = [], [], []
    step_actions = []
    done = False
    step = 0

    while not done:
        with torch.no_grad():
            t_inter = torch.from_numpy(_normalize_obs(obs["inter_feat"][0], "inter_feat", obs_stats)).float().to(device)
            t_intra = torch.from_numpy(_normalize_obs(obs["intra_feat"][0], "intra_feat", obs_stats)).float().to(device)
            t_glob = torch.from_numpy(_normalize_obs(obs["global_feat"][0], "global_feat", obs_stats)).float().to(device)

            history["inter"].append(t_inter)
            history["intra"].append(t_intra)
            history["global"].append(t_glob)

            rtg_raw = target_rtg - cumulative_reward
            rtg_val = np.sign(rtg_raw) * np.log1p(np.abs(rtg_raw))
            history["rtg"].append(torch.tensor([rtg_val], dtype=torch.float32, device=device))
            history["timesteps"].append(torch.tensor([step], dtype=torch.long, device=device))

            if len(history["rtg"]) > context_len:
                for k in history:
                    history[k].pop(0)

            in_inter = pad_stack_tensor(history["inter"], context_len, None, device)
            in_intra = pad_stack_tensor(history["intra"], context_len, None, device)
            in_global = pad_stack_tensor(history["global"], context_len, 2, device)
            in_rtg = pad_stack_tensor(history["rtg"], context_len, 1, device)
            in_steps = pad_stack_tensor(history["timesteps"], context_len, 1, device).squeeze(-1).long()

            act_seq = torch.stack(history["actions"])
            if act_seq.shape[0] < context_len:
                pad = torch.zeros((context_len - act_seq.shape[0], len(action_dims)), dtype=torch.long, device=device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_actions = act_seq.unsqueeze(0)

            states_in = {"inter": in_inter, "intra": in_intra, "global": in_global}
            real_len = min(len(history["rtg"]), context_len)
            mask = torch.zeros((1, context_len), device=device)
            mask[0, -real_len:] = 1.0

            preds = model(states=states_in, actions=in_actions, returns=in_rtg, timesteps=in_steps, attention_mask=mask)
            logits = preds[0, -1, :]

            pred_actions = []
            start = 0
            for ds in action_dims:
                pred_actions.append(torch.argmax(logits[start:start + ds]).item())
                start += ds

        action_tensor = torch.tensor(pred_actions, dtype=torch.long, device=device)
        history["actions"].append(action_tensor)
        step_actions.append(pred_actions)

        env_action = np.array([pred_actions], dtype=np.int32)
        next_obs, rewards, dones, infos = env.step(env_action)

        cumulative_reward += float(rewards[0])
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        hd, nd, hv, nv, ha, na = _compute_step_metrics(info)
        step_hp_dists.append(hd / ha if ha > 0 else 0.0)
        step_nhp_dists.append(nd / na if na > 0 else 0.0)
        step_rewards.append(float(rewards[0]))
        hp_dist_sum += hd; nhp_dist_sum += nd
        hp_viol_sum += hv; nhp_viol_sum += nv
        hp_active_t += ha; nhp_active_t += na

        obs = next_obs
        step += 1
        if dones[0]:
            done = True

    hp_v = hp_viol_sum / hp_active_t if hp_active_t > 0 else 0.0
    nhp_v = nhp_viol_sum / nhp_active_t if nhp_active_t > 0 else 0.0
    hp_d = hp_dist_sum / hp_active_t if hp_active_t > 0 else 0.0
    nhp_d = nhp_dist_sum / nhp_active_t if nhp_active_t > 0 else 0.0
    return {
        "hp_v": hp_v, "nhp_v": nhp_v, "hp_d": hp_d, "nhp_d": nhp_d,
        "ep_reward": cumulative_reward,
        "step_hp_dists": step_hp_dists,
        "step_nhp_dists": step_nhp_dists,
        "step_rewards": step_rewards,
        "step_actions": step_actions,
    }


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default="data/channel_generality/dt_v2/dataset/metadata.json")
    parser.add_argument("--target_rtg", type=float, default=0.0)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--init_episode", type=int, default=None)
    parser.add_argument("--max_episode", type=int, default=None)
    parser.add_argument("--save_root", type=str, default="data/channel_generality/dt_v2/dt_test_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--encoder", type=str, default="cross_attn",
        choices=["cross_attn", "mlp", "slice_attn"],
        help="State encoder type used by the trained checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(2)

    obs_stats = _load_obs_stats(args.meta_path)

    # Action dims for v2: codebook(11) + ordering(5x5) + intra(5x3)
    action_dims = [11] + [5] * NUM_SLICES + [3] * NUM_SLICES
    act_dim = sum(action_dims)

    context_len = 20
    embed_dim = 512

    model_cfg = OmegaConf.create({
        "context_len": context_len, "embed_dim": embed_dim,
        "n_layer": 12, "n_head": 16, "activation": "relu",
        "dropout": 0.1, "act_dim": act_dim,
    })
    model = build_dt_v2(
        model_cfg=model_cfg, action_dims=action_dims,
        inter_dim=INTER_DIM, intra_dim=INTRA_DIM, embed_dim=embed_dim,
        encoder_type=args.encoder,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")
    print(f"Action dims: {action_dims} (total {act_dim})")

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = PathContext(os.getcwd())

    all_results = {}
    for scen in args.scenarios:
        print(f"\n--- Scenario {scen} ---")
        scen_hp, scen_nhp = [], []
        scen_hp_d, scen_nhp_d = [], []

        for seed in args.seeds:
            cfg = env_cfg.env_settings.copy()
            cfg.mode = "testing"
            cfg.scenario_mode = "inside"
            cfg.inside.testing.active_scenario_list = [scen]
            if args.init_episode is not None:
                cfg.inside.testing.init_scenario_episode = args.init_episode
            if args.max_episode is not None:
                cfg.inside.testing.max_scenario_episodes = args.max_episode

            def _make(c=cfg, s=seed):
                def _init():
                    return HierarchicalSlicingEnvV2(c, np.random.default_rng(s), pm)
                return _init

            env = DummyVecEnv([_make()])
            ep_rewards_list = []
            all_step_hp_dists = []
            all_step_nhp_dists = []
            all_step_actions = []
            seed_hp_viols = []
            seed_nhp_viols = []
            for ep in range(args.n_episodes):
                ep_result = run_dt_episode(
                    env, model, device, args.target_rtg, context_len, action_dims, obs_stats
                )
                hp_v, nhp_v = ep_result["hp_v"], ep_result["nhp_v"]
                hp_d, nhp_d = ep_result["hp_d"], ep_result["nhp_d"]
                scen_hp.append(hp_v)
                scen_nhp.append(nhp_v)
                scen_hp_d.append(hp_d)
                scen_nhp_d.append(nhp_d)
                seed_hp_viols.append(hp_v)
                seed_nhp_viols.append(nhp_v)
                ep_rewards_list.append(ep_result["ep_reward"])
                all_step_hp_dists.extend(ep_result["step_hp_dists"])
                all_step_nhp_dists.extend(ep_result["step_nhp_dists"])
                all_step_actions.extend(ep_result["step_actions"])
                print(f"  seed{seed} ep{ep}: hp_v={hp_v:.4f} nhp_v={nhp_v:.4f} hp_d={hp_d:.4f} nhp_d={nhp_d:.4f}")
            env.close()
            raw_dir = os.path.join(args.save_root, "metric_raw", f"scenario_{scen}")
            os.makedirs(raw_dir, exist_ok=True)
            np.savez(
                os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                ep_rewards=np.array(ep_rewards_list, dtype=np.float64),
                hp_viols=np.array(seed_hp_viols, dtype=np.float64),
                nhp_viols=np.array(seed_nhp_viols, dtype=np.float64),
                step_hp_dist=np.array(all_step_hp_dists, dtype=np.float64),
                step_nhp_dist=np.array(all_step_nhp_dists, dtype=np.float64),
                actions=np.array(all_step_actions, dtype=np.int32),
            )

        summary = {
            "hp_viol_mean": float(np.mean(scen_hp)),
            "nhp_viol_mean": float(np.mean(scen_nhp)),
            "hp_dist_mean": float(np.mean(scen_hp_d)),
            "nhp_dist_mean": float(np.mean(scen_nhp_d)),
            "combined": float(np.mean(scen_hp) + np.mean(scen_nhp)),
        }
        all_results[scen] = summary
        print(f"  S{scen}: hp_v={summary['hp_viol_mean']:.4f} nhp_v={summary['nhp_viol_mean']:.4f} "
              f"hp_d={summary['hp_dist_mean']:.4f} nhp_d={summary['nhp_dist_mean']:.4f} "
              f"comb={summary['combined']:.4f}")

        _save_json(summary, os.path.join(args.save_root, f"scenario_{scen}", "summary.json"))

    total = sum(r["combined"] for r in all_results.values())
    print(f"\nTOTAL: {total:.4f}")
    _save_json({"per_scenario": {str(k): v for k, v in all_results.items()}, "total": total},
               os.path.join(args.save_root, "global_summary.json"))


def test_dt_v2(cfg, path_context):
    """Hydra entry point: called by channel_generality.py → test_dt_v2 mode.

    Uses cfg.test_dt_v2.  Episode range defaults from env_ha.yaml "testing"
    unless test_dt_v2.init_episode / max_episode are set.
    """
    import torch
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv

    tc = cfg.test_dt_v2
    device = torch.device(str(tc.device))
    torch.set_num_threads(2)

    obs_stats = _load_obs_stats(str(tc.meta_path))

    action_dims = [11] + [5] * NUM_SLICES + [3] * NUM_SLICES
    act_dim = sum(action_dims)
    context_len = int(tc.model.context_len)
    embed_dim = int(tc.model.embed_dim)

    model_cfg = OmegaConf.create({
        "context_len": context_len,
        "embed_dim": embed_dim,
        "n_layer": int(tc.model.n_layer),
        "n_head": int(tc.model.n_head),
        "activation": str(tc.model.activation),
        "dropout": float(tc.model.dropout),
        "act_dim": act_dim,
    })
    model = build_dt_v2(
        model_cfg=model_cfg,
        action_dims=action_dims,
        inter_dim=INTER_DIM,
        intra_dim=INTRA_DIM,
        embed_dim=embed_dim,
        encoder_type=str(tc.encoder_type),
        encoder_hidden_dim=int(tc.model.encoder_hidden_dim) if tc.model.get("encoder_hidden_dim") is not None else None,
        encoder_num_heads=int(tc.model.encoder_num_heads) if tc.model.get("encoder_num_heads") is not None else None,
    ).to(device)
    model.load_state_dict(torch.load(str(tc.model_path), map_location=device))
    model.eval()
    print(f"[test_dt_v2] model loaded from {tc.model_path}")

    env_cfg_raw = OmegaConf.load("conf/environment/env_ha.yaml")
    pm = path_context

    all_results = {}
    for scen in list(tc.test_scenarios):
        scen_hp, scen_nhp, scen_hp_d, scen_nhp_d = [], [], [], []
        for seed in list(tc.test_seeds):
            env_settings = env_cfg_raw.env_settings.copy()
            env_settings.mode = str(tc.env_updates.mode)  # "testing" → ep 60-79
            env_settings.scenario_mode = "inside"
            env_settings.inside.testing.active_scenario_list = [scen]
            if tc.get("init_episode") is not None:
                env_settings.inside.testing.init_scenario_episode = int(tc.init_episode)
            if tc.get("max_episode") is not None:
                env_settings.inside.testing.max_scenario_episodes = int(tc.max_episode)

            def _make(s=env_settings, sd=seed):
                def _init():
                    return HierarchicalSlicingEnvV2(s, np.random.default_rng(sd), pm)
                return _init

            env = DummyVecEnv([_make()])
            ep_rewards_list = []
            all_step_hp_dists = []
            all_step_nhp_dists = []
            all_step_actions = []
            seed_hp_viols = []
            seed_nhp_viols = []
            for _ in range(int(tc.n_episodes)):
                ep_result = run_dt_episode(
                    env, model, device, float(tc.target_rtg),
                    context_len, action_dims, obs_stats,
                )
                hp_v, nhp_v = ep_result["hp_v"], ep_result["nhp_v"]
                hp_d, nhp_d = ep_result["hp_d"], ep_result["nhp_d"]
                scen_hp.append(hp_v)
                scen_nhp.append(nhp_v)
                scen_hp_d.append(hp_d)
                scen_nhp_d.append(nhp_d)
                seed_hp_viols.append(hp_v)
                seed_nhp_viols.append(nhp_v)
                ep_rewards_list.append(ep_result["ep_reward"])
                all_step_hp_dists.extend(ep_result["step_hp_dists"])
                all_step_nhp_dists.extend(ep_result["step_nhp_dists"])
                all_step_actions.extend(ep_result["step_actions"])
            env.close()
            raw_dir = os.path.join(str(tc.save_root), "metric_raw", f"scenario_{scen}")
            os.makedirs(raw_dir, exist_ok=True)
            np.savez(
                os.path.join(raw_dir, f"ep_seed{seed}.npz"),
                ep_rewards=np.array(ep_rewards_list, dtype=np.float64),
                hp_viols=np.array(seed_hp_viols, dtype=np.float64),
                nhp_viols=np.array(seed_nhp_viols, dtype=np.float64),
                step_hp_dist=np.array(all_step_hp_dists, dtype=np.float64),
                step_nhp_dist=np.array(all_step_nhp_dists, dtype=np.float64),
                actions=np.array(all_step_actions, dtype=np.int32),
            )

        summary = {
            "hp_viol_mean": float(np.mean(scen_hp)),
            "nhp_viol_mean": float(np.mean(scen_nhp)),
            "hp_dist_mean": float(np.mean(scen_hp_d)),
            "nhp_dist_mean": float(np.mean(scen_nhp_d)),
            "combined": float(np.mean(scen_hp) + np.mean(scen_nhp)),
        }
        all_results[scen] = summary
        print(f"  S{scen}: hp_v={summary['hp_viol_mean']:.4f} nhp_v={summary['nhp_viol_mean']:.4f} "
              f"hp_d={summary['hp_dist_mean']:.4f} nhp_d={summary['nhp_dist_mean']:.4f} "
              f"comb={summary['combined']:.4f}")
        _save_json(summary, os.path.join(str(tc.save_root), f"scenario_{scen}", "summary.json"))

    total = sum(r["combined"] for r in all_results.values())
    print(f"\n[test_dt_v2] TOTAL combined: {total:.4f}")
    _save_json(
        {"per_scenario": {str(k): v for k, v in all_results.items()}, "total": total},
        os.path.join(str(tc.save_root), "global_summary.json"),
    )


def test_dt_v2_tiny(cfg, path_context):
    """Hydra entry point: called by channel_generality.py → test_dt_v2_tiny mode."""
    # Reuse the exact same evaluation pipeline; tiny mode is pure config routing.
    class _CfgProxy:
        pass

    cfg_proxy = _CfgProxy()
    cfg_proxy.test_dt_v2 = cfg.test_dt_v2_tiny
    return test_dt_v2(cfg_proxy, path_context)


if __name__ == "__main__":
    main()
