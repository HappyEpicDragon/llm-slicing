"""
Test DT-baseline model on ppo-baseline's environment (env_sb3.py).

DT-baseline uses the same obs/action space as ppo-baseline:
  - Obs: Dict per agent → flattened to 145-dim vector
  - Action: player_0 continuous (5,) + player_1..5 discrete (3 each)

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -m src.basic_apis.dt_v2.test_dt_baseline --model_path <path>
"""
import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.basic_apis.dt_v2.model_baseline import build_dt_baseline
from src.basic_apis.ppo.ppo_baseline.env_ray import env_creator

NUM_SLICES = 5
CONT_DIM = 5
DISC_DIMS = [3, 3, 3, 3, 3]
OBS_DIM = 145


def _load_obs_stats(meta_path):
    if not meta_path or not os.path.exists(meta_path):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    obs_stats_raw = meta.get("obs_stats", {})
    flat_stats = obs_stats_raw.get("flat", None)
    if flat_stats is None:
        return None, None
    mean = np.array(flat_stats["mean"], dtype=np.float32)
    std = np.array(flat_stats["std"], dtype=np.float32) + 1e-6
    return mean, std


def _normalize_obs(flat_obs, obs_mean, obs_std):
    if obs_mean is None:
        return flat_obs
    normed = (flat_obs - obs_mean) / obs_std
    return np.clip(normed, -5.0, 5.0).astype(np.float32)


def flatten_marl_observation(obs_dict):
    flat_obs = []
    if "player_0" in obs_dict:
        p0 = obs_dict["player_0"]
        flat_obs.append(p0["observations"] if isinstance(p0, dict) and "observations" in p0 else p0)
    for i in range(1, 6):
        key = f"player_{i}"
        if key in obs_dict:
            p = obs_dict[key]
            flat_obs.append(p["observations"] if isinstance(p, dict) and "observations" in p else p)
    return np.concatenate(flat_obs, axis=0).astype(np.float32)


def _compute_step_metrics(info, num_slices=NUM_SLICES):
    flat_info = info if isinstance(info, dict) else {}
    if flat_info and not any(k.startswith("drift/") or k.startswith("meta/") for k in flat_info):
        for v in flat_info.values():
            if isinstance(v, dict) and any(k.startswith("drift/") or k.startswith("meta/") for k in v):
                flat_info = v
                break

    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = flat_info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(flat_info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if flat_info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        mean_drift = float(np.mean(drifts))
        if mean_drift < 0:
            if is_hp:
                hp_dist += mean_drift
                hp_viols += 1
            else:
                nhp_dist += mean_drift
                nhp_viols += 1
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def run_dt_baseline_episode(env, model, device, target_rtg, context_len,
                            obs_mean, obs_std):
    obs_dict, _ = env.reset()

    act_dim = CONT_DIM + len(DISC_DIMS)
    dummy_action = torch.zeros(act_dim, device=device)
    history = {"obs": [], "actions": [dummy_action], "rtg": [], "timesteps": []}

    cumulative_reward = 0.0
    hp_viol_sum = nhp_viol_sum = 0
    hp_dist_sum = nhp_dist_sum = 0.0
    hp_active_t = nhp_active_t = 0
    step_hp_dists, step_nhp_dists, step_rewards = [], [], []
    done = False
    step = 0

    def _pad_stack(lst):
        seq = torch.stack(lst)
        if seq.ndim == 1:
            seq = seq.unsqueeze(-1)
        curr = seq.shape[0]
        if curr < context_len:
            pad_shape = list(seq.shape)
            pad_shape[0] = context_len - curr
            seq = torch.cat([torch.zeros(pad_shape, device=device), seq], dim=0)
        return seq.unsqueeze(0)

    while not done:
        with torch.no_grad():
            flat_obs = flatten_marl_observation(obs_dict)
            norm_obs = _normalize_obs(flat_obs, obs_mean, obs_std)
            t_obs = torch.from_numpy(norm_obs).float().to(device)
            history["obs"].append(t_obs)

            rtg_raw = target_rtg - cumulative_reward
            rtg_val = np.sign(rtg_raw) * np.log1p(np.abs(rtg_raw))
            history["rtg"].append(torch.tensor([rtg_val], dtype=torch.float32, device=device))
            history["timesteps"].append(torch.tensor([step], dtype=torch.long, device=device))

            if len(history["rtg"]) > context_len:
                for k in history:
                    history[k].pop(0)

            in_obs = _pad_stack(history["obs"])
            in_rtg = _pad_stack(history["rtg"])
            in_time = _pad_stack(history["timesteps"]).squeeze(-1).long()

            act_seq = torch.stack(history["actions"])
            if act_seq.shape[0] < context_len:
                pad = torch.zeros((context_len - act_seq.shape[0], act_dim), device=device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_actions = act_seq.unsqueeze(0)

            real_len = min(len(history["rtg"]), context_len)
            mask = torch.zeros((1, context_len), device=device)
            mask[0, -real_len:] = 1.0

            preds = model(states=in_obs, actions=in_actions, returns=in_rtg,
                          timesteps=in_time, attention_mask=mask)

            cont_preds = preds["cont_preds"][0, -1, :]
            disc_logits = preds["disc_preds"][0, -1, :]

            cont_action = torch.clamp(cont_preds, -1.0, 1.0).cpu().numpy()
            disc_actions = []
            offset = 0
            for d in DISC_DIMS:
                disc_actions.append(torch.argmax(disc_logits[offset:offset + d]).item())
                offset += d

            action_dict = {"player_0": cont_action}
            for i in range(NUM_SLICES):
                action_dict[f"player_{i + 1}"] = disc_actions[i]

            hybrid_act = torch.cat([
                cont_preds,
                torch.tensor(disc_actions, dtype=torch.float32, device=device)
            ])
            history["actions"].append(hybrid_act)

        obs_dict_new, rewards, terminated, truncated, infos = env.step(action_dict)
        done_flag = terminated.get("__all__", False) or truncated.get("__all__", False)

        r = rewards.get("player_0", 0.0)
        cumulative_reward += r
        step_rewards.append(r)

        step_info = infos.get("__common__", infos)
        hd, nd, hv, nv, ha, na = _compute_step_metrics(step_info)
        step_hp_dists.append(hd / ha if ha > 0 else 0.0)
        step_nhp_dists.append(nd / na if na > 0 else 0.0)
        hp_dist_sum += hd
        nhp_dist_sum += nd
        hp_viol_sum += hv
        nhp_viol_sum += nv
        hp_active_t += ha
        nhp_active_t += na

        obs_dict = obs_dict_new
        step += 1
        if done_flag:
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
    }


def _save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _build_env_config(cfg_environment, scenario_id, paths_cfg=None, workdir=None, mode="testing",
                      init_episode=None, max_episode=None):
    env_config = OmegaConf.to_container(cfg_environment, resolve=True)
    scenario_mode = env_config.get("scenario_mode", "inside")
    env_config["mode"] = mode
    env_config["scenario_mode"] = scenario_mode
    env_config[scenario_mode][mode]["active_scenario_list"] = [scenario_id]
    if init_episode is not None:
        env_config[scenario_mode][mode]["initial_episode"] = init_episode
    if max_episode is not None:
        env_config[scenario_mode][mode]["max_episode"] = max_episode
    env_config["paths_cfg"] = paths_cfg
    env_config["workdir"] = workdir
    return env_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--meta_path", type=str, default="data/channel_generality/dt_baseline_v2/dataset/metadata.json")
    parser.add_argument("--target_rtg", type=float, default=0.0)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--init_episode", type=int, default=None)
    parser.add_argument("--max_episode", type=int, default=None)
    parser.add_argument("--save_root", type=str, default="data/channel_generality/dt_baseline_v2/eval_ood")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.set_num_threads(2)

    obs_mean, obs_std = _load_obs_stats(args.meta_path)

    model_cfg = OmegaConf.create({
        "context_len": args.context_len,
        "embed_dim": args.embed_dim,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "activation": "relu",
        "dropout": 0.1,
    })
    model = build_dt_baseline(
        model_cfg=model_cfg, obs_dim=OBS_DIM,
        cont_dim=CONT_DIM, disc_dims=DISC_DIMS, embed_dim=args.embed_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    env_cfg = OmegaConf.load("conf/environment/env_ha.yaml")

    all_results = {}
    for scen in args.scenarios:
        print(f"\n--- Scenario {scen} ---")
        scen_hp, scen_nhp, scen_hp_d, scen_nhp_d = [], [], [], []

        for seed in args.seeds:
            env_config = OmegaConf.to_container(env_cfg, resolve=True)
            env_settings = env_config.get("env_settings", env_config)
            scenario_mode = env_settings.get("scenario_mode", "inside")
            env_settings["mode"] = "testing"
            env_settings["scenario_mode"] = scenario_mode
            env_settings[scenario_mode]["testing"]["active_scenario_list"] = [scen]
            if args.init_episode is not None:
                env_settings[scenario_mode]["testing"]["initial_episode"] = args.init_episode
            if args.max_episode is not None:
                env_settings[scenario_mode]["testing"]["max_episode"] = args.max_episode
            env_settings["paths_cfg"] = None
            env_settings["workdir"] = os.getcwd()
            env_settings["seed"] = seed

            env = env_creator(env_settings)

            ep_rewards_list = []
            all_step_hp_dists, all_step_nhp_dists = [], []
            seed_hp_viols, seed_nhp_viols = [], []
            for ep in range(args.n_episodes):
                ep_result = run_dt_baseline_episode(
                    env, model, device, args.target_rtg, args.context_len,
                    obs_mean, obs_std,
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
                print(f"  seed{seed} ep{ep}: hp_v={hp_v:.4f} nhp_v={nhp_v:.4f} "
                      f"hp_d={hp_d:.4f} nhp_d={nhp_d:.4f}")

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
              f"hp_d={summary['hp_dist_mean']:.4f} nhp_d={summary['nhp_dist_mean']:.4f}")
        _save_json(summary, os.path.join(args.save_root, f"scenario_{scen}", "summary.json"))

    total = sum(r["combined"] for r in all_results.values())
    print(f"\nTOTAL: {total:.4f}")
    _save_json({"per_scenario": {str(k): v for k, v in all_results.items()}, "total": total},
               os.path.join(args.save_root, "global_summary.json"))


def test_dt_baseline(cfg, paths_cfg=None, workdir=None):
    """Hydra entry point: called by channel_generality.py → test_dt_baseline mode."""
    tc = cfg.test_dt_baseline
    paths_cfg = paths_cfg if paths_cfg is not None else cfg.get("paths", None)
    workdir = workdir if workdir is not None else str(cfg.get("workdir", os.getcwd()))
    device = torch.device(str(tc.device))
    torch.set_num_threads(2)

    obs_mean, obs_std = _load_obs_stats(str(tc.meta_path))

    context_len = int(tc.model.context_len)
    embed_dim = int(tc.model.embed_dim)

    model_cfg = OmegaConf.create({
        "context_len": context_len,
        "embed_dim": embed_dim,
        "n_layer": int(tc.model.n_layer),
        "n_head": int(tc.model.n_head),
        "activation": str(tc.model.get("activation", "relu")),
        "dropout": float(tc.model.get("dropout", 0.1)),
    })
    model = build_dt_baseline(
        model_cfg=model_cfg, obs_dim=OBS_DIM,
        cont_dim=CONT_DIM, disc_dims=DISC_DIMS, embed_dim=embed_dim,
    ).to(device)
    model.load_state_dict(torch.load(str(tc.model_path), map_location=device))
    model.eval()
    print(f"[test_dt_baseline] model loaded from {tc.model_path}")

    env_cfg_raw = OmegaConf.to_container(tc.environment, resolve=True)
    all_results = {}
    for scen in list(tc.test_scenarios):
        scen_hp, scen_nhp, scen_hp_d, scen_nhp_d = [], [], [], []
        for seed in list(tc.test_seeds):
            import copy
            env_config = copy.deepcopy(env_cfg_raw)
            scenario_mode = env_config.get("scenario_mode", "inside")
            env_config["mode"] = "testing"
            env_config["scenario_mode"] = scenario_mode
            env_config[scenario_mode]["testing"]["active_scenario_list"] = [scen]
            if tc.get("init_episode") is not None:
                env_config[scenario_mode]["testing"]["initial_episode"] = int(tc.init_episode)
            if tc.get("max_episode") is not None:
                env_config[scenario_mode]["testing"]["max_episode"] = int(tc.max_episode)
            env_config["paths_cfg"] = paths_cfg
            env_config["workdir"] = workdir
            env_config["seed"] = seed

            env = env_creator(env_config)

            ep_rewards_list = []
            all_step_hp_dists, all_step_nhp_dists = [], []
            seed_hp_viols, seed_nhp_viols = [], []
            for _ in range(int(tc.n_episodes)):
                ep_result = run_dt_baseline_episode(
                    env, model, device, float(tc.target_rtg),
                    context_len, obs_mean, obs_std,
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
              f"hp_d={summary['hp_dist_mean']:.4f} nhp_d={summary['nhp_dist_mean']:.4f}")
        _save_json(summary, os.path.join(str(tc.save_root), f"scenario_{scen}", "summary.json"))

    total = sum(r["combined"] for r in all_results.values())
    print(f"\n[test_dt_baseline] TOTAL combined: {total:.4f}")
    _save_json(
        {"per_scenario": {str(k): v for k, v in all_results.items()}, "total": total},
        os.path.join(str(tc.save_root), "global_summary.json"),
    )


if __name__ == "__main__":
    main()
