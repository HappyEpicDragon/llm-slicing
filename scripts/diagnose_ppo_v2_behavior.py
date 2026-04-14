"""
诊断脚本：分析 PPO-v2 MLP 模型的行为模式。
1. 确认 S0 joint == per-scenario 是否为 bug（打印实际动作轨迹）
2. 检查是否收敛为常数策略（动作的方差/熵）
3. 在恶化场景（S2/S3/S4）对比 joint vs per-scenario expert 的行为差异
"""
import os, sys, argparse
import numpy as np
from collections import Counter
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

NUM_SLICES = 5
# 动作维度：[codebook(11), order0(5), order1(5), order2(5), order3(5), order4(5),
#            intra0(3), intra1(3), intra2(3), intra3(3), intra4(3)]
ACTION_NAMES = (
    ["codebook"] +
    [f"order_{i}" for i in range(5)] +
    [f"intra_{i}" for i in range(5)]
)
INTRA_LABELS = {0: "RR", 1: "PF", 2: "MaxThr"}


def collect_trajectory(model, scenario, seed, n_episodes, env_cfg, pm, label=""):
    """运行 n_episodes，收集每步的完整动作向量和关键 info 字段。"""
    env_settings = env_cfg.env_settings.copy()
    env_settings.mode = "testing"
    env_settings.scenario_mode = "inside"
    env_settings.inside.testing.active_scenario_list = [scenario]

    def _make(s=env_settings, sd=seed):
        def _init():
            return HierarchicalSlicingEnvV2(s, np.random.default_rng(sd), pm)
        return _init

    env = DummyVecEnv([_make()])
    all_actions = []      # shape (total_steps, 11)
    all_rewards = []
    all_hp_drifts = []
    all_nhp_drifts = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            all_actions.append(action[0].copy())
            all_rewards.append(float(rewards[0]))

            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            for s_idx in range(NUM_SLICES):
                for m in ("thr", "rel", "lat"):
                    if info.get(f"meta/slice_{s_idx}_{m}_req", 0) > 0:
                        d = info.get(f"drift/slice_{s_idx}_{m}", 0.0)
                        prio = info.get(f"meta/slice_{s_idx}_priority", 0)
                        if prio > 0:
                            all_hp_drifts.append(d)
                        else:
                            all_nhp_drifts.append(d)
            if dones[0]:
                done = True

    env.close()
    actions = np.array(all_actions)  # (T, 11)
    return actions, np.array(all_rewards), np.array(all_hp_drifts), np.array(all_nhp_drifts)


def analyze_actions(actions, label):
    """打印动作的分布统计，判断是否常数策略。"""
    print(f"\n  [{label}] 动作分布统计 (T={len(actions)} steps)")
    print(f"  {'Dim':<12} {'Most-common':>12} {'Freq%':>8} {'Unique':>8} {'Std':>8}")
    print(f"  {'-'*52}")

    is_constant = True
    for dim_i, name in enumerate(ACTION_NAMES):
        vals = actions[:, dim_i]
        counter = Counter(vals.tolist())
        most_common_val, most_common_cnt = counter.most_common(1)[0]
        freq_pct = 100.0 * most_common_cnt / len(vals)
        n_unique = len(counter)
        std = np.std(vals)
        flag = " ← CONSTANT" if n_unique == 1 else (" ← near-constant" if freq_pct > 95 else "")
        if n_unique > 1:
            is_constant = False
        print(f"  {name:<12} {int(most_common_val):>12} {freq_pct:>7.1f}% {n_unique:>8} {std:>8.3f}{flag}")

    if is_constant:
        print(f"\n  ⚠️  CONSTANT POLICY: model always outputs the SAME action!")
    return is_constant


def compare_behavior(actions_expert, actions_joint, scenario):
    """对比 expert 和 joint 的动作差异。"""
    print(f"\n  [S{scenario}] Expert vs Joint 动作对比")
    print(f"  {'Dim':<12} {'Expert mode':>14} {'Joint mode':>14} {'Agreement%':>12} {'Δ mean':>10}")
    print(f"  {'-'*60}")

    for dim_i, name in enumerate(ACTION_NAMES):
        T = min(len(actions_expert), len(actions_joint))
        e = actions_expert[:T, dim_i]
        j = actions_joint[:T, dim_i]
        agree_pct = 100.0 * np.mean(e == j)
        e_mode = Counter(e.tolist()).most_common(1)[0][0]
        j_mode = Counter(j.tolist()).most_common(1)[0][0]
        delta = np.mean(j.astype(float)) - np.mean(e.astype(float))
        flag = " ←" if abs(delta) > 0.3 or agree_pct < 50 else ""
        print(f"  {name:<12} {int(e_mode):>14} {int(j_mode):>14} {agree_pct:>11.1f}% {delta:>+10.3f}{flag}")


def print_drift_summary(hp_drifts, nhp_drifts, label):
    """打印 drift 统计（负值=违规）。"""
    print(f"\n  [{label}] Drift 统计")
    if len(hp_drifts) > 0:
        hp_viol_rate = np.mean(hp_drifts < 0)
        print(f"    HP:  mean={np.mean(hp_drifts):+.4f}  viol_rate={hp_viol_rate:.3f}  "
              f"min={np.min(hp_drifts):+.4f}  max={np.max(hp_drifts):+.4f}")
    if len(nhp_drifts) > 0:
        nhp_viol_rate = np.mean(nhp_drifts < 0)
        print(f"    NHP: mean={np.mean(nhp_drifts):+.4f}  viol_rate={nhp_viol_rate:.3f}  "
              f"min={np.min(nhp_drifts):+.4f}  max={np.max(nhp_drifts):+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/channel_generality/dt_v2_mean_rwd")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)

    # ------------------------------------------------------------------ #
    # 1. 确认 S0 joint == per-scenario 是否 bug                          #
    # ------------------------------------------------------------------ #
    print("=" * 70)
    print("1. 验证 S0: per-scenario expert vs joint — 动作是否相同")
    print("=" * 70)

    m_s0 = PPO.load(os.path.join(ROOT, args.root, "ppo_s0/best_model/best_model.zip"), device=args.device)
    m_joint = PPO.load(os.path.join(ROOT, args.root, "ppo_joint_mlp/best_model/best_model.zip"), device=args.device)

    act_s0_on_s0, rwd_s0, hp_s0, nhp_s0 = collect_trajectory(m_s0, 0, args.seed, args.n_episodes, env_cfg, pm)
    act_joint_on_s0, rwd_j0, hp_j0, nhp_j0 = collect_trajectory(m_joint, 0, args.seed, args.n_episodes, env_cfg, pm)

    T = min(len(act_s0_on_s0), len(act_joint_on_s0))
    action_agreement_s0 = np.mean(act_s0_on_s0[:T] == act_joint_on_s0[:T])
    print(f"\n  S0 全维度动作一致率: {action_agreement_s0*100:.1f}%")
    print(f"  Mean reward — expert: {np.mean(rwd_s0):.4f}   joint: {np.mean(rwd_j0):.4f}")
    compare_behavior(act_s0_on_s0, act_joint_on_s0, 0)
    print_drift_summary(hp_s0, nhp_s0, "S0 expert")
    print_drift_summary(hp_j0, nhp_j0, "S0 joint")

    # ------------------------------------------------------------------ #
    # 2. 常数策略检查（S0、S2 各跑 expert 和 joint）                      #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("2. 常数策略检查")
    print("=" * 70)

    analyze_actions(act_s0_on_s0, "ppo_s0 on S0")
    analyze_actions(act_joint_on_s0, "joint_mlp on S0")

    m_s2 = PPO.load(os.path.join(ROOT, args.root, "ppo_s2/best_model/best_model.zip"), device=args.device)
    act_s2_on_s2, _, _, _ = collect_trajectory(m_s2, 2, args.seed, args.n_episodes, env_cfg, pm)
    act_joint_on_s2, _, _, _ = collect_trajectory(m_joint, 2, args.seed, args.n_episodes, env_cfg, pm)
    analyze_actions(act_s2_on_s2, "ppo_s2 on S2")
    analyze_actions(act_joint_on_s2, "joint_mlp on S2")

    # ------------------------------------------------------------------ #
    # 3. 恶化场景行为对比（S2、S3、S4）                                   #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("3. 恶化场景行为对比 (joint vs per-scenario expert)")
    print("=" * 70)

    for scen, model_name in [(2, "ppo_s2"), (3, "ppo_s3"), (4, "ppo_s4")]:
        print(f"\n{'─'*70}")
        print(f"  Scenario {scen}")

        m_expert = PPO.load(os.path.join(ROOT, args.root, f"{model_name}/best_model/best_model.zip"), device=args.device)
        act_e, _, hp_e, nhp_e = collect_trajectory(m_expert, scen, args.seed, args.n_episodes, env_cfg, pm)
        act_j, _, hp_j, nhp_j = collect_trajectory(m_joint,  scen, args.seed, args.n_episodes, env_cfg, pm)

        print_drift_summary(hp_e, nhp_e, f"S{scen} expert")
        print_drift_summary(hp_j, nhp_j, f"S{scen} joint")
        compare_behavior(act_e, act_j, scen)

        # 额外：codebook 分布
        print(f"\n  [S{scen}] Codebook 选择分布")
        e_cb = Counter(act_e[:, 0].tolist())
        j_cb = Counter(act_j[:, 0].tolist())
        all_codes = sorted(set(e_cb) | set(j_cb))
        print(f"  {'Code':>6} {'Expert%':>10} {'Joint%':>10}")
        for c in all_codes:
            ep = 100.0 * e_cb.get(c, 0) / len(act_e)
            jp = 100.0 * j_cb.get(c, 0) / len(act_j)
            flag = " ←" if abs(ep - jp) > 10 else ""
            print(f"  {c:>6} {ep:>9.1f}% {jp:>9.1f}%{flag}")

        # 额外：ordering 对比（谁排最前/最后）
        print(f"\n  [S{scen}] Ordering 均值 (order_i 越大 = 排越靠前)")
        for i in range(5):
            e_mean = np.mean(act_e[:, 1+i])
            j_mean = np.mean(act_j[:, 1+i])
            delta = j_mean - e_mean
            flag = " ←" if abs(delta) > 0.5 else ""
            print(f"    Slice {i}: expert={e_mean:.2f}  joint={j_mean:.2f}  Δ={delta:+.2f}{flag}")

        # 额外：intra-scheduler 分布
        print(f"\n  [S{scen}] Intra-scheduler (per slice)")
        for i in range(5):
            e_mode = Counter(act_e[:, 6+i].tolist()).most_common(1)[0][0]
            j_mode = Counter(act_j[:, 6+i].tolist()).most_common(1)[0][0]
            flag = " ←" if e_mode != j_mode else ""
            print(f"    Slice {i}: expert={INTRA_LABELS[e_mode]}({e_mode})  "
                  f"joint={INTRA_LABELS[j_mode]}({j_mode}){flag}")


if __name__ == "__main__":
    main()
