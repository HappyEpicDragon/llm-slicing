"""
评测 PPO-v2 联合模型在 s5-s9（OOD 场景）上的性能，与 ppo_multi 基线对比。

Episode 范围：ep 0-19（与 ppo_multi 评估协议一致，覆盖完全未见场景）。
模型：ppo_joint_mlp / ppo_joint_slice_attn
"""
import os, sys, json, argparse
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

NUM_SLICES = 5


def _compute_step_metrics(info):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(NUM_SLICES):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [
            float(info.get(f"drift/slice_{s_idx}_{m}", 0.0))
            for m in ("thr", "rel", "lat")
            if info.get(f"meta/slice_{s_idx}_{m}_req", 0) > 0
        ]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        mean_drift = float(np.mean(drifts))
        if mean_drift < 0:
            if is_hp:
                hp_viols += 1; hp_dist += mean_drift
            else:
                nhp_viols += 1; nhp_dist += mean_drift
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def eval_model_on_scenario(model_path, scenario, n_episodes, seeds, device, env_cfg, pm):
    """评测单个模型在单个 OOD 场景（s5-s9）上的性能，ep 0-(n_episodes-1)。"""
    model = PPO.load(model_path, device=device)
    all_hp_v, all_nhp_v, all_hp_d, all_nhp_d = [], [], [], []

    for seed in seeds:
        # 使用 testing 模式作为基础，覆盖 episode 范围为 ep 0-19（与 ppo_multi 一致）
        env_settings = env_cfg.env_settings.copy()
        env_settings.mode = "testing"
        env_settings.scenario_mode = "inside"
        env_settings.inside.testing.active_scenario_list = [scenario]
        env_settings.inside.testing.init_scenario_episode = 0
        env_settings.inside.testing.max_scenario_episodes = n_episodes  # ep 0-19

        def _make(s=env_settings, sd=seed):
            def _init():
                return HierarchicalSlicingEnvV2(s, np.random.default_rng(sd), pm)
            return _init

        env = DummyVecEnv([_make()])
        ep_hp_v, ep_nhp_v, ep_hp_d, ep_nhp_d = [], [], [], []

        for ep in range(n_episodes):
            hd = nd = hv = nv = ha = na = 0
            obs = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                _hd, _nd, _hv, _nv, _ha, _na = _compute_step_metrics(info)
                hd += _hd; nd += _nd; hv += _hv; nv += _nv; ha += _ha; na += _na
                if dones[0]:
                    done = True
            ep_hp_v.append(hv / ha if ha > 0 else 0.0)
            ep_nhp_v.append(nv / na if na > 0 else 0.0)
            ep_hp_d.append(hd / ha if ha > 0 else 0.0)
            ep_nhp_d.append(nd / na if na > 0 else 0.0)

        all_hp_v.extend(ep_hp_v)
        all_nhp_v.extend(ep_nhp_v)
        all_hp_d.extend(ep_hp_d)
        all_nhp_d.extend(ep_nhp_d)
        env.close()

    return {
        "hp_viol_mean":  float(np.mean(all_hp_v)),
        "nhp_viol_mean": float(np.mean(all_nhp_v)),
        "hp_dist_mean":  float(np.mean(all_hp_d)),
        "nhp_dist_mean": float(np.mean(all_nhp_d)),
        "combined":      float(np.mean(all_hp_v) + np.mean(all_nhp_v)),
        "episodes": len(all_hp_v),
    }


def load_ppo_multi_metrics(base_path, scenarios):
    """从已有 metric_json 读取 ppo_multi 基线指标（seed_0）。"""
    results = {}
    for sc in scenarios:
        d = os.path.join(base_path, f"scenario_{sc}", "seed_0")
        try:
            hpv  = json.load(open(f"{d}/hp_violations.json"))["mean"]
            nhpv = json.load(open(f"{d}/nhp_violations.json"))["mean"]
            hpd  = json.load(open(f"{d}/hp_distance.json"))["mean"]
            nhpd = json.load(open(f"{d}/nhp_distance.json"))["mean"]
            results[sc] = {
                "hp_viol_mean": hpv, "nhp_viol_mean": nhpv,
                "hp_dist_mean": hpd, "nhp_dist_mean": nhpd,
                "combined": hpv + nhpv,
            }
        except FileNotFoundError as e:
            print(f"  [WARN] ppo_multi S{sc}: {e}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/channel_generality/dt_v2_mean_rwd")
    parser.add_argument("--ppo_multi_root",
                        default="data/channel_generality/ppo_multi/metric_json")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="与 ppo_multi 一致，使用 ep 0-19")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device_mlp",   default="cuda:0")
    parser.add_argument("--device_attn",  default="cuda:1")
    parser.add_argument("--save_root",
                        default="data/channel_generality/dt_v2_mean_rwd/eval_results")
    parser.add_argument("--out_file", default="joint_ood_comparison.json",
                        help="输出 JSON 文件名")
    args = parser.parse_args()

    OOD_SCENARIOS = list(range(5, 10))
    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    print(f"OOD eval (s5-s9): n_episodes={args.n_episodes}, seeds={args.seeds}")
    print(f"Episode range: ep 0-{args.n_episodes-1} (matches ppo_multi protocol)")
    print("=" * 78)

    # ---------- ppo_multi 基线 ----------
    print("\n[Baseline] ppo_multi (pre-computed metrics)")
    ppo_multi = load_ppo_multi_metrics(os.path.join(ROOT, args.ppo_multi_root), OOD_SCENARIOS)
    for sc, r in ppo_multi.items():
        print(f"  S{sc}: hp_v={r['hp_viol_mean']:.4f}  nhp_v={r['nhp_viol_mean']:.4f}  "
              f"hp_d={r['hp_dist_mean']:.4f}  nhp_d={r['nhp_dist_mean']:.4f}  comb={r['combined']:.4f}")
    if ppo_multi:
        print(f"  → mean combined: {np.mean([v['combined'] for v in ppo_multi.values()]):.4f}")

    # ---------- ppo_joint_mlp ----------
    print("\n[Joint MLP] ppo_joint_mlp → s5-s9")
    mlp_path = os.path.join(ROOT, args.root, "ppo_joint_mlp", "best_model", "best_model.zip")
    mlp_results = {}
    if os.path.isfile(mlp_path):
        for sc in OOD_SCENARIOS:
            r = eval_model_on_scenario(mlp_path, sc, args.n_episodes, args.seeds,
                                       args.device_mlp, env_cfg, pm)
            mlp_results[sc] = r
            print(f"  S{sc}: hp_v={r['hp_viol_mean']:.4f}  nhp_v={r['nhp_viol_mean']:.4f}  "
                  f"hp_d={r['hp_dist_mean']:.4f}  nhp_d={r['nhp_dist_mean']:.4f}  comb={r['combined']:.4f}")
        print(f"  → mean combined: {np.mean([v['combined'] for v in mlp_results.values()]):.4f}")
    else:
        print(f"  SKIP (not found: {mlp_path})")

    # ---------- ppo_joint_slice_attn ----------
    print("\n[Joint SliceAttn] ppo_joint_slice_attn → s5-s9")
    attn_path = os.path.join(ROOT, args.root, "ppo_joint_slice_attn", "best_model", "best_model.zip")
    attn_results = {}
    if os.path.isfile(attn_path):
        for sc in OOD_SCENARIOS:
            r = eval_model_on_scenario(attn_path, sc, args.n_episodes, args.seeds,
                                       args.device_attn, env_cfg, pm)
            attn_results[sc] = r
            print(f"  S{sc}: hp_v={r['hp_viol_mean']:.4f}  nhp_v={r['nhp_viol_mean']:.4f}  "
                  f"hp_d={r['hp_dist_mean']:.4f}  nhp_d={r['nhp_dist_mean']:.4f}  comb={r['combined']:.4f}")
        print(f"  → mean combined: {np.mean([v['combined'] for v in attn_results.values()]):.4f}")
    else:
        print(f"  SKIP (not found: {attn_path})")

    # ---------- 保存结果 ----------
    os.makedirs(args.save_root, exist_ok=True)
    summary = {
        "ppo_multi":          {str(k): v for k, v in ppo_multi.items()},
        "ppo_joint_mlp":      {str(k): v for k, v in mlp_results.items()},
        "ppo_joint_slice_attn": {str(k): v for k, v in attn_results.items()},
        "config": {
            "n_episodes": args.n_episodes,
            "seeds": args.seeds,
            "episode_range": f"ep 0-{args.n_episodes-1} (matches ppo_multi)",
            "ood_scenarios": OOD_SCENARIOS,
        },
    }
    out_path = os.path.join(args.save_root, args.out_file)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ---------- 汇总对比表 ----------
    print("\n" + "=" * 86)
    print(f"{'Model':<28} {'S5':>8} {'S6':>8} {'S7':>8} {'S8':>8} {'S9':>8} {'Mean':>8}")
    print("-" * 86)
    for label, res in [("ppo_multi", ppo_multi),
                        ("ppo_joint_mlp", mlp_results),
                        ("ppo_joint_slice_attn", attn_results)]:
        if not res:
            continue
        vals = [res.get(sc, {}).get("combined", float("nan")) for sc in OOD_SCENARIOS]
        mean_v = np.nanmean(vals)
        row = "  ".join(f"{v:>6.4f}" for v in vals)
        print(f"{label:<28} {row}  {mean_v:>6.4f}")
    print("=" * 86)


if __name__ == "__main__":
    main()
