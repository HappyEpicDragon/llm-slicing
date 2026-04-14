"""
PPO-MLP (dt_v2) 决策瓶颈诊断：4 个实验合一
  Exp1: Codebook 选择分布 + 实际 PRB 分配比例
  Exp2: 排序动作稳定性 + sort_slices 启发式消融
  Exp3: Intra-slice 调度器选择分布
  Exp4: S2 step-level 违约溯源
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

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
CODEBOOK_TEMPLATES = [
    [27, 27, 27, 27, 27],
    [30, 28, 27, 26, 24],
    [31, 29, 27, 25, 23],
    [33, 29, 27, 24, 22],
    [36, 30, 26, 23, 20],
    [39, 31, 26, 22, 17],
    [44, 32, 25, 19, 15],
    [51, 32, 22, 17, 13],
    [60, 31, 19, 14, 11],
    [70, 28, 16, 11, 10],
    [79, 23, 13, 10, 10],
]


def compute_sort_slices_heuristic(inner_env):
    """Replicate PPO-baseline sort_slices: argsort(ues_per_slice * traffic_req) ascending."""
    slice_assoc = inner_env.components.slices.ue_assoc
    ues_per_slice = np.sum(slice_assoc, axis=1)
    raw = inner_env.last_raw_obs
    if raw is None:
        return np.arange(NUM_SLICES)
    slice_req = raw.get("slice_req", {})
    traffic = np.zeros(NUM_SLICES)
    for s in range(NUM_SLICES):
        req = slice_req.get(f"slice_{s}", {})
        ues_info = req.get("ues", {})
        traffic[s] = float(ues_info.get("traffic", 0.0))
    total = ues_per_slice * traffic
    return np.argsort(total)


def compute_step_metrics(info):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(NUM_SLICES):
        is_hp = info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [
            float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
            for met in ("thr", "rel", "lat")
            if info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0
        ]
        if not drifts:
            continue
        if is_hp:
            hp_active += 1
        else:
            nhp_active += 1
        if min(drifts) < 0:
            if is_hp:
                hp_viols += 1
                hp_dist += min(d for d in drifts if d < 0)
            else:
                nhp_viols += 1
                nhp_dist += min(d for d in drifts if d < 0)
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def make_env(env_cfg, scen, n_episodes, seed=0):
    cfg = env_cfg.env_settings.copy()
    cfg.mode = "testing"
    cfg.scenario_mode = "inside"
    cfg.inside.testing.active_scenario_list = [scen]
    cfg.inside.testing.init_scenario_episode = 0
    cfg.inside.testing.max_scenario_episodes = n_episodes
    pm = PathManager(ROOT)

    def _init():
        return HierarchicalSlicingEnvV2(cfg, np.random.default_rng(seed), pm)
    return DummyVecEnv([_init])


def run_diagnostic_episodes(env, model, n_episodes, collect_step_detail=False):
    """Run episodes and collect per-step data."""
    inner = env.envs[0]
    all_ep_data = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_data = {
            "codebook_indices": [],
            "ordering_actions": [],
            "sorted_slice_indices": [],
            "intra_modes": [],
            "inter_alloc_ratios": [],
            "rescue_rbs": [],
            "step_violations": [],
            "heuristic_orderings": [],
        }
        if collect_step_detail:
            ep_data["step_details"] = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            raw_action = action[0] if action.ndim > 1 else action

            ep_data["codebook_indices"].append(int(raw_action[0]))
            ep_data["ordering_actions"].append(raw_action[1:1 + NUM_SLICES].tolist())
            ep_data["intra_modes"].append(raw_action[1 + NUM_SLICES:].tolist())

            next_obs, _, dones, infos = env.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos

            ep_data["sorted_slice_indices"].append(inner.last_sorted_slice_indices.tolist())
            ep_data["inter_alloc_ratios"].append(inner.last_inter_alloc_ratio.tolist())
            ep_data["rescue_rbs"].append(inner.last_rescue_rbs)
            ep_data["heuristic_orderings"].append(compute_sort_slices_heuristic(inner).tolist())

            hp_d, nhp_d, hv, nv, ha, na = compute_step_metrics(info)
            ep_data["step_violations"].append({
                "hp_dist": hp_d, "nhp_dist": nhp_d,
                "hp_viols": hv, "nhp_viols": nv,
                "hp_active": ha, "nhp_active": na,
            })

            if collect_step_detail:
                detail = {"step": len(ep_data["step_details"])}
                for s in range(NUM_SLICES):
                    detail[f"s{s}_priority"] = info.get(f"meta/slice_{s}_priority", 0)
                    for met in ("thr", "rel", "lat"):
                        detail[f"s{s}_{met}_drift"] = info.get(f"drift/slice_{s}_{met}", 0.0)
                detail["rescue_rbs"] = inner.last_rescue_rbs
                detail["codebook_idx"] = int(raw_action[0])
                detail["inter_quotas"] = (inner.last_inter_alloc_ratio * inner.alloc_unit_count).astype(int).tolist()
                detail["ordering"] = inner.last_sorted_slice_indices.tolist()
                ep_data["step_details"].append(detail)

            obs = next_obs
            if dones[0]:
                done = True

        all_ep_data.append(ep_data)
    return all_ep_data


def run_ablation_heuristic_ordering(env, model, n_episodes):
    """Ablation: replace learned ordering with sort_slices heuristic."""
    inner = env.envs[0]
    ep_metrics = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        hp_viol_sum = nhp_viol_sum = 0
        hp_active_t = nhp_active_t = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            heuristic_order = compute_sort_slices_heuristic(inner)
            inner.set_forced_ordering(heuristic_order)

            next_obs, _, dones, infos = env.step(action)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            _, _, hv, nv, ha, na = compute_step_metrics(info)
            hp_viol_sum += hv
            nhp_viol_sum += nv
            hp_active_t += ha
            nhp_active_t += na

            obs = next_obs
            if dones[0]:
                done = True

        inner.set_forced_ordering(None)
        hp_v = hp_viol_sum / hp_active_t if hp_active_t > 0 else 0.0
        nhp_v = nhp_viol_sum / nhp_active_t if nhp_active_t > 0 else 0.0
        ep_metrics.append((hp_v, nhp_v))

    return ep_metrics


# ===================== Analysis / Print =====================

def analyze_exp1(all_data, scen):
    """Codebook distribution + PRB allocation ratios."""
    print(f"\n{'='*60}")
    print(f"  实验1: S{scen} Codebook 选择分布 + PRB 分配比例")
    print(f"{'='*60}")

    all_cb = []
    all_ratios = []
    for ep in all_data:
        all_cb.extend(ep["codebook_indices"])
        all_ratios.extend(ep["inter_alloc_ratios"])

    cb_counts = Counter(all_cb)
    total_steps = len(all_cb)
    print(f"\n  总步数: {total_steps}")
    print(f"\n  Codebook 使用频率:")
    print(f"  {'idx':>4} {'template':>25} {'count':>7} {'freq%':>7}")
    for idx in range(len(CODEBOOK_TEMPLATES)):
        cnt = cb_counts.get(idx, 0)
        tpl = CODEBOOK_TEMPLATES[idx]
        print(f"  {idx:>4} {str(tpl):>25} {cnt:>7} {100.0*cnt/total_steps:>6.1f}%")

    unique_used = sum(1 for v in cb_counts.values() if v > 0)
    top1_idx = cb_counts.most_common(1)[0][0]
    top1_pct = 100.0 * cb_counts.most_common(1)[0][1] / total_steps
    print(f"\n  使用的不同 codebook 数: {unique_used}/11")
    print(f"  最常用 codebook: idx={top1_idx} ({top1_pct:.1f}%)")

    ratios = np.array(all_ratios)
    print(f"\n  各切片 PRB 配额比例 (mean ± std):")
    print(f"  {'slice':>6} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
    for s in range(NUM_SLICES):
        m = np.mean(ratios[:, s])
        sd = np.std(ratios[:, s])
        mn = np.min(ratios[:, s])
        mx = np.max(ratios[:, s])
        print(f"  {s:>6} {m:>8.4f} {sd:>8.4f} {mn:>8.4f} {mx:>8.4f}")


def analyze_exp2(all_data, scen):
    """Ordering stability + heuristic comparison."""
    print(f"\n{'='*60}")
    print(f"  实验2: S{scen} 排序动作稳定性 & sort_slices 对比")
    print(f"{'='*60}")

    all_orderings = []
    all_heuristic = []
    match_count = 0
    total_steps = 0

    for ep in all_data:
        for learned, heur in zip(ep["sorted_slice_indices"], ep["heuristic_orderings"]):
            all_orderings.append(tuple(learned))
            all_heuristic.append(tuple(heur))
            if tuple(learned) == tuple(heur):
                match_count += 1
            total_steps += 1

    order_counts = Counter(all_orderings)
    print(f"\n  总步数: {total_steps}")
    print(f"  学习到的排序 Top-5:")
    for rank, (order, cnt) in enumerate(order_counts.most_common(5), 1):
        print(f"    #{rank}: {list(order)}  ({cnt} 次, {100.0*cnt/total_steps:.1f}%)")

    heur_counts = Counter(all_heuristic)
    print(f"\n  sort_slices 启发式排序 Top-3:")
    for rank, (order, cnt) in enumerate(heur_counts.most_common(3), 1):
        print(f"    #{rank}: {list(order)}  ({cnt} 次, {100.0*cnt/total_steps:.1f}%)")

    print(f"\n  学习排序与启发式完全一致的步比例: {100.0*match_count/total_steps:.1f}%")


def analyze_exp3(all_data, scen, slice_priorities):
    """Intra-slice scheduler distribution."""
    print(f"\n{'='*60}")
    print(f"  实验3: S{scen} Intra-slice 调度器选择分布")
    print(f"{'='*60}")

    sched_counts = defaultdict(lambda: [0, 0, 0])
    total_steps = 0
    for ep in all_data:
        for modes in ep["intra_modes"]:
            total_steps += 1
            for s in range(NUM_SLICES):
                sched_counts[s][int(modes[s])] += 1

    names = ["RR", "PF", "MT"]
    print(f"\n  {'场景':>4} {'切片':>4} {'HP?':>4} {'RR%':>7} {'PF%':>7} {'MT%':>7}")
    for s in range(NUM_SLICES):
        hp = "HP" if slice_priorities.get(s, 0) > 0 else "NHP"
        counts = sched_counts[s]
        total = sum(counts)
        pcts = [100.0 * c / total if total > 0 else 0 for c in counts]
        print(f"  S{scen:>3} {s:>4} {hp:>4} {pcts[0]:>6.1f}% {pcts[1]:>6.1f}% {pcts[2]:>6.1f}%")


def analyze_exp4(ep_data, scen):
    """S2 step-level violation tracing."""
    print(f"\n{'='*60}")
    print(f"  实验4: S{scen} Step-Level 违约溯源 (Episode 0)")
    print(f"{'='*60}")

    details = ep_data[0]["step_details"]
    total_steps = len(details)

    violation_steps = []
    for d in details:
        violating = []
        for s in range(NUM_SLICES):
            drifts = []
            for met in ("thr", "rel", "lat"):
                val = d.get(f"s{s}_{met}_drift", 0.0)
                if abs(val) > 1e-6 or val < 0:
                    drifts.append((met, val))
            min_drift = min((v for _, v in drifts), default=0.0)
            if min_drift < 0:
                prio = d[f"s{s}_priority"]
                violating.append((s, "HP" if prio > 0 else "NHP", min_drift, drifts))
        if violating:
            violation_steps.append((d["step"], violating, d))

    print(f"\n  总步数: {total_steps}, 有违约的步数: {len(violation_steps)}")

    # Per-slice violation statistics
    slice_viol_count = defaultdict(int)
    slice_viol_drift_sum = defaultdict(float)
    for _, viols, _ in violation_steps:
        for s, ptype, min_d, _ in viols:
            slice_viol_count[(s, ptype)] += 1
            slice_viol_drift_sum[(s, ptype)] += min_d

    print(f"\n  各切片违约统计:")
    print(f"  {'切片':>4} {'类型':>4} {'违约步数':>8} {'占比%':>7} {'平均drift':>10}")
    for s in range(NUM_SLICES):
        for ptype in ("HP", "NHP"):
            cnt = slice_viol_count.get((s, ptype), 0)
            if cnt > 0:
                avg_d = slice_viol_drift_sum[(s, ptype)] / cnt
                print(f"  {s:>4} {ptype:>4} {cnt:>8} {100.0*cnt/total_steps:>6.1f}% {avg_d:>10.4f}")

    # Rescue statistics
    rescue_rbs = [d["rescue_rbs"] for _, _, d in violation_steps]
    all_rescue = [d["rescue_rbs"] for d in details]
    print(f"\n  Rescue RB 统计:")
    print(f"    触发 rescue 的步比例: {100.0*sum(1 for r in all_rescue if r > 0)/total_steps:.1f}%")
    print(f"    rescue RB 均值: {np.mean(all_rescue):.1f}, 最大: {np.max(all_rescue)}")

    # Codebook at violation steps
    viol_cb = [d["codebook_idx"] for _, _, d in violation_steps]
    cb_counts = Counter(viol_cb)
    print(f"\n  违约步使用的 codebook 分布:")
    for idx, cnt in cb_counts.most_common(5):
        print(f"    idx={idx} {CODEBOOK_TEMPLATES[idx]}: {cnt} 次")

    # Show first 10 violation steps in detail
    print(f"\n  前 10 个违约步详情:")
    print(f"  {'step':>5} {'cb':>3} {'quotas':>25} {'ordering':>20} {'rescue':>6} {'违约切片':>30}")
    for step_idx, viols, d in violation_steps[:10]:
        viol_str = ", ".join([f"s{s}({pt},{md:.3f})" for s, pt, md, _ in viols])
        print(f"  {step_idx:>5} {d['codebook_idx']:>3} {str(d['inter_quotas']):>25} "
              f"{str(d['ordering']):>20} {d['rescue_rbs']:>6} {viol_str}")

    # Root cause analysis
    print(f"\n  === 根因分析 ===")

    cb_at_viol = Counter(d["codebook_idx"] for _, _, d in violation_steps)
    cb_all = Counter(d["codebook_idx"] for d in details)

    viol_with_rescue = sum(1 for _, _, d in violation_steps if d["rescue_rbs"] > 0)
    print(f"  - 违约步中 rescue 触发率: {100.0*viol_with_rescue/max(len(violation_steps),1):.1f}%")

    ordering_at_viol = Counter(tuple(d["ordering"]) for _, _, d in violation_steps)
    ordering_all = Counter(tuple(d["ordering"]) for d in details)
    print(f"  - 违约步最常见排序: {list(ordering_at_viol.most_common(1)[0][0]) if ordering_at_viol else 'N/A'}")
    print(f"  - 全局最常见排序:   {list(ordering_all.most_common(1)[0][0]) if ordering_all else 'N/A'}")

    hp_viol_steps = sum(1 for _, viols, _ in violation_steps
                        for s, pt, _, _ in viols if pt == "HP")
    nhp_viol_steps = sum(1 for _, viols, _ in violation_steps
                         for s, pt, _, _ in viols if pt == "NHP")
    print(f"  - HP 违约次数: {hp_viol_steps}, NHP 违约次数: {nhp_viol_steps}")

    avg_quota_viol = np.mean([d["inter_quotas"] for _, _, d in violation_steps], axis=0) if violation_steps else np.zeros(5)
    avg_quota_all = np.mean([d["inter_quotas"] for d in details], axis=0)
    print(f"  - 违约步 avg PRB quotas: {np.round(avg_quota_viol, 1).tolist()}")
    print(f"  - 全局 avg PRB quotas:   {np.round(avg_quota_all, 1).tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/channel_generality/dt_v2_mlp_ent05")
    parser.add_argument("--scenarios", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n_episodes", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    print(f"PPO-MLP 决策瓶颈诊断: scenarios={args.scenarios}, n_episodes={args.n_episodes}")
    print(f"模型根路径: {args.root}")

    all_scenario_data = {}

    # ==================== Collect data ====================
    for scen in args.scenarios:
        model_path = os.path.join(ROOT, args.root, f"ppo_s{scen}", "best_model", "best_model.zip")
        if not os.path.isfile(model_path):
            print(f"\nS{scen}: SKIP (missing {model_path})")
            continue
        print(f"\n--- S{scen}: Loading {model_path} ---", flush=True)
        model = PPO.load(model_path, device=args.device)

        collect_detail = (scen == 2)
        env = make_env(env_cfg, scen, args.n_episodes)
        try:
            data = run_diagnostic_episodes(env, model, args.n_episodes, collect_step_detail=collect_detail)
        finally:
            env.close()
        all_scenario_data[scen] = data

    # ==================== Exp 1: Codebook ====================
    print("\n" + "#" * 70)
    print("#  实验1: Codebook 选择分布 + PRB 分配比例")
    print("#" * 70)
    for scen, data in sorted(all_scenario_data.items()):
        analyze_exp1(data, scen)

    # ==================== Exp 2: Ordering ====================
    print("\n" + "#" * 70)
    print("#  实验2: 排序动作稳定性 + sort_slices 对比")
    print("#" * 70)
    for scen, data in sorted(all_scenario_data.items()):
        analyze_exp2(data, scen)

    # Ablation: replace ordering with heuristic
    print(f"\n{'='*60}")
    print(f"  实验2-消融: 用 sort_slices 启发式替换学习到的排序")
    print(f"{'='*60}")
    ablation_results = {}
    for scen in sorted(all_scenario_data.keys()):
        model_path = os.path.join(ROOT, args.root, f"ppo_s{scen}", "best_model", "best_model.zip")
        model = PPO.load(model_path, device=args.device)
        env = make_env(env_cfg, scen, args.n_episodes)
        try:
            ab_metrics = run_ablation_heuristic_ordering(env, model, args.n_episodes)
        finally:
            env.close()
        hp_m = np.mean([m[0] for m in ab_metrics])
        nhp_m = np.mean([m[1] for m in ab_metrics])
        ablation_results[scen] = (hp_m, nhp_m, hp_m + nhp_m)

    # Also compute original metrics for comparison
    original_results = {}
    for scen, data in sorted(all_scenario_data.items()):
        all_hp = []
        all_nhp = []
        for ep in data:
            ep_hp = sum(s["hp_viols"] for s in ep["step_violations"])
            ep_ha = sum(s["hp_active"] for s in ep["step_violations"])
            ep_nhp = sum(s["nhp_viols"] for s in ep["step_violations"])
            ep_na = sum(s["nhp_active"] for s in ep["step_violations"])
            all_hp.append(ep_hp / ep_ha if ep_ha > 0 else 0)
            all_nhp.append(ep_nhp / ep_na if ep_na > 0 else 0)
        original_results[scen] = (np.mean(all_hp), np.mean(all_nhp), np.mean(all_hp) + np.mean(all_nhp))

    print(f"\n  {'场景':>4} {'原始comb':>10} {'消融comb':>10} {'Δcomb':>10} {'原始hp':>10} {'消融hp':>10} {'原始nhp':>10} {'消融nhp':>10}")
    for scen in sorted(all_scenario_data.keys()):
        o = original_results[scen]
        a = ablation_results[scen]
        delta = a[2] - o[2]
        print(f"  S{scen:>3} {o[2]:>10.4f} {a[2]:>10.4f} {delta:>+10.4f} {o[0]:>10.4f} {a[0]:>10.4f} {o[1]:>10.4f} {a[1]:>10.4f}")

    mean_orig = np.mean([v[2] for v in original_results.values()])
    mean_abl = np.mean([v[2] for v in ablation_results.values()])
    print(f"\n  Mean comb: 原始={mean_orig:.4f}, 消融={mean_abl:.4f}, Δ={mean_abl - mean_orig:+.4f}")

    # ==================== Exp 3: Intra ====================
    print("\n" + "#" * 70)
    print("#  实验3: Intra-slice 调度器选择分布")
    print("#" * 70)
    for scen, data in sorted(all_scenario_data.items()):
        # Determine priorities from the collected data
        priorities = {}
        for ep in data:
            for v in ep["step_violations"]:
                pass
        # Get priorities from step_detail or re-derive
        env = make_env(env_cfg, scen, 1)
        inner = env.envs[0]
        obs = env.reset()
        model_path = os.path.join(ROOT, args.root, f"ppo_s{scen}", "best_model", "best_model.zip")
        m = PPO.load(model_path, device=args.device)
        action, _ = m.predict(obs, deterministic=True)
        _, _, _, infos = env.step(action)
        info = infos[0]
        for s in range(NUM_SLICES):
            priorities[s] = info.get(f"meta/slice_{s}_priority", 0)
        env.close()
        analyze_exp3(data, scen, priorities)

    # ==================== Exp 4: S2 Step-Level ====================
    if 2 in all_scenario_data:
        print("\n" + "#" * 70)
        print("#  实验4: S2 Step-Level 违约溯源")
        print("#" * 70)
        analyze_exp4(all_scenario_data[2], 2)

    print("\n" + "=" * 70)
    print("诊断完成")


if __name__ == "__main__":
    main()
