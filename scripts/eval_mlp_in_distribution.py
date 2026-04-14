"""
评测 PPO-v2 MLP 模型在各自训练场景上的性能（in-distribution）。
- ppo_s0 → s0, ppo_s1 → s1, ..., ppo_s4 → s4
- ppo_joint_mlp → s0, s1, s2, s3, s4

使用 held-out 测试集：ep 60-79（testing 模式），不覆盖 init_scenario_episode。
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
    """评测单个模型在单个场景上的性能（held-out ep 60-79）。"""
    model = PPO.load(model_path, device=device)
    all_hp_v, all_nhp_v, all_hp_d, all_nhp_d = [], [], [], []

    for seed in seeds:
        env_settings = env_cfg.env_settings.copy()
        env_settings.mode = "testing"          # → ep 60-79
        env_settings.scenario_mode = "inside"
        env_settings.inside.testing.active_scenario_list = [scenario]
        # 不覆盖 init_scenario_episode，使用 YAML 默认 (init=60, max=80)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/channel_generality/dt_v2_mean_rwd")
    parser.add_argument("--suffix", default="",
                        help="模型目录后缀：'' for MLP, '_slice_attn' for SliceAttn")
    parser.add_argument("--joint_name", default="",
                        help="joint 模型目录名（默认根据 suffix 自动推断）")
    parser.add_argument("--out_key", default="",
                        help="保存 JSON 时的键名前缀（默认根据 suffix 自动推断）")
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--save_root", default="data/channel_generality/dt_v2_mean_rwd/eval_results")
    parser.add_argument("--out_file", default="",
                        help="输出 JSON 文件名（默认根据 suffix 推断）")
    args = parser.parse_args()

    suffix = args.suffix  # e.g. "" or "_slice_attn"
    arch_label = "SliceAttn" if "slice_attn" in suffix else "MLP"
    joint_dir = args.joint_name or f"ppo_joint{suffix if suffix else '_mlp'}"
    out_key_per = f"per_scenario_expert{suffix.replace('_', ' ').strip() or '_mlp'}".replace(" ", "_")
    out_key_joint = joint_dir
    out_file = args.out_file or f"{arch_label.lower()}_in_distribution.json"

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    print(f"In-distribution eval [{arch_label}]: n_episodes={args.n_episodes}, seeds={args.seeds}")
    print(f"Suffix: '{suffix}'  Joint dir: '{joint_dir}'")
    print(f"Episode range: held-out ep 60-79 (testing mode)")
    print("=" * 70)

    # --- Per-scenario 专家模型 ---
    print(f"\n[Per-Scenario Experts{' '+arch_label if suffix else ''}] ppo_s0{suffix} → s0, ...")
    per_scenario_results = {}
    for sid in range(5):
        model_path = os.path.join(ROOT, args.root, f"ppo_s{sid}{suffix}", "best_model", "best_model.zip")
        if not os.path.isfile(model_path):
            print(f"  S{sid}: SKIP (not found: {model_path})")
            continue
        r = eval_model_on_scenario(model_path, sid, args.n_episodes, args.seeds, args.device, env_cfg, pm)
        per_scenario_results[sid] = r
        print(f"  S{sid}: hp_v={r['hp_viol_mean']:.4f}  nhp_v={r['nhp_viol_mean']:.4f}  "
              f"hp_d={r['hp_dist_mean']:.4f}  nhp_d={r['nhp_dist_mean']:.4f}  comb={r['combined']:.4f}")

    if per_scenario_results:
        mean_comb = np.mean([v["combined"] for v in per_scenario_results.values()])
        print(f"  → mean combined (s0-s4): {mean_comb:.4f}")

    # --- Joint 模型在 s0-s4 上 ---
    print(f"\n[Joint {arch_label}] {joint_dir} → s0-s4")
    joint_model_path = os.path.join(ROOT, args.root, joint_dir, "best_model", "best_model.zip")
    joint_results = {}
    if os.path.isfile(joint_model_path):
        for sid in range(5):
            r = eval_model_on_scenario(joint_model_path, sid, args.n_episodes, args.seeds, args.device, env_cfg, pm)
            joint_results[sid] = r
            print(f"  S{sid}: hp_v={r['hp_viol_mean']:.4f}  nhp_v={r['nhp_viol_mean']:.4f}  "
                  f"hp_d={r['hp_dist_mean']:.4f}  nhp_d={r['nhp_dist_mean']:.4f}  comb={r['combined']:.4f}")
        mean_comb_joint = np.mean([v["combined"] for v in joint_results.values()])
        print(f"  → mean combined (s0-s4): {mean_comb_joint:.4f}")
    else:
        print(f"  SKIP (not found: {joint_model_path})")

    # --- 保存结果 ---
    os.makedirs(args.save_root, exist_ok=True)
    summary = {
        out_key_per:   {str(k): v for k, v in per_scenario_results.items()},
        out_key_joint: {str(k): v for k, v in joint_results.items()},
        "config": {"n_episodes": args.n_episodes, "seeds": args.seeds,
                   "suffix": suffix, "episode_range": "testing mode (ep 60-79)"},
    }
    out_path = os.path.join(args.save_root, out_file)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # --- 汇总表格 ---
    print("\n" + "=" * 70)
    print(f"{'Model':<28} {'Scen':>4} {'hp_v':>8} {'nhp_v':>8} {'hp_d':>10} {'nhp_d':>10} {'comb':>8}")
    print("-" * 78)
    for sid, r in per_scenario_results.items():
        m = f"ppo_s{sid}{suffix}"
        print(f"{m:<28} {sid:>4} {r['hp_viol_mean']:>8.4f} {r['nhp_viol_mean']:>8.4f} "
              f"{r['hp_dist_mean']:>10.4f} {r['nhp_dist_mean']:>10.4f} {r['combined']:>8.4f}")
    print("-" * 78)
    for sid, r in joint_results.items():
        print(f"{joint_dir:<28} {sid:>4} {r['hp_viol_mean']:>8.4f} {r['nhp_viol_mean']:>8.4f} "
              f"{r['hp_dist_mean']:>10.4f} {r['nhp_dist_mean']:>10.4f} {r['combined']:>8.4f}")


if __name__ == "__main__":
    main()
