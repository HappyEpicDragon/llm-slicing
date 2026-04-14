"""
Oracle ordering comparison: Quantify the contribution of ordering vs codebook.

Experiments (all use intra=RR):
  1. Constant ordering sweep: 120 orderings × best CB → find best constant ordering
  2. Greedy ordering oracle: per-step best ordering (120 options) × best CB
  3. Combined greedy: per-step best (CB × ordering) = 11×120 = 1320 options

Metric: comb = hp_distance + nhp_distance (closer to 0 = better)

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/oracle_ordering_comparison.py --scenario 0
"""
import argparse, os, sys, time, copy
from itertools import permutations, product
import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

NUM_SLICES = 5
ALL_ORDERINGS = list(permutations(range(NUM_SLICES)))  # 120 permutations
ALL_INTRAS = list(product(range(3), repeat=NUM_SLICES))  # 243 combinations


def make_env(scenario_id, seed=0):
    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)
    cfg = env_cfg.env_settings.copy()
    cfg.mode = "testing"
    cfg.scenario_mode = "inside"
    cfg.inside.testing.active_scenario_list = [scenario_id]
    cfg.inside.testing.init_scenario_episode = 0
    cfg.inside.testing.max_scenario_episodes = 20
    return HierarchicalSlicingEnvV2(cfg, np.random.default_rng(seed), pm)


def compute_step_metrics(info):
    hp_dist = nhp_dist = 0.0
    hp_active = nhp_active = 0
    for s_idx in range(NUM_SLICES):
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
        neg = [d for d in drifts if d < 0]
        if neg:
            if is_hp:
                hp_dist += min(neg)
            else:
                nhp_dist += min(neg)
    return hp_dist, nhp_dist, hp_active, nhp_active


def make_action(codebook, ordering, intra=None, num_slices=5):
    action = np.zeros(1 + num_slices + num_slices, dtype=int)
    action[0] = codebook
    action[1:1 + num_slices] = ordering
    if intra is not None:
        action[1 + num_slices:] = intra
    return action


def sample_intra_candidates(sample_intras, seed=42):
    rr = ALL_INTRAS[0]
    if sample_intras <= 1:
        return [rr]
    if sample_intras >= len(ALL_INTRAS):
        return ALL_INTRAS

    rng = np.random.default_rng(seed)
    remaining = np.arange(1, len(ALL_INTRAS))
    picked = rng.choice(remaining, size=sample_intras - 1, replace=False)
    return [rr] + [ALL_INTRAS[i] for i in picked]


def run_episode(env, codebook, ordering, intra=None, max_steps=1000):
    """Run one full episode with fixed codebook, ordering and intra policy."""
    obs, _ = env.reset()
    step_hp, step_nhp = [], []
    hp_active_total = nhp_active_total = 0
    rewards = []

    for t in range(max_steps):
        env.set_forced_ordering(np.array(ordering, dtype=int))
        action = make_action(codebook, ordering, intra=intra)
        obs, r, done, _, info = env.step(action)
        rewards.append(r)
        hp_d, nhp_d, hp_a, nhp_a = compute_step_metrics(info)
        step_hp.append(hp_d)
        step_nhp.append(nhp_d)
        hp_active_total += hp_a
        nhp_active_total += nhp_a
        if done:
            break

    hp_distance = sum(step_hp) / max(hp_active_total, 1)
    nhp_distance = sum(step_nhp) / max(nhp_active_total, 1)
    return hp_distance + nhp_distance, hp_distance, nhp_distance, np.mean(rewards)


# ── Experiment 1: Constant ordering sweep ─────────────────────────────

def constant_ordering_sweep(scenario_id, best_cb, seed=0, max_steps=1000):
    print("\n" + "=" * 70)
    print(f"EXP 1: Constant Ordering Sweep (120 orderings, CB={best_cb}, intra=RR)")
    print("=" * 70)

    results = []
    for i, ordering in enumerate(ALL_ORDERINGS):
        env = make_env(scenario_id, seed=seed)
        comb, hp, nhp, rew = run_episode(env, best_cb, ordering, max_steps)
        env.close()
        results.append((ordering, comb, hp, nhp, rew))
        if (i + 1) % 20 == 0:
            print(f"  Tested {i + 1}/120 orderings...", flush=True)

    results.sort(key=lambda x: x[1], reverse=True)  # closest to 0 first

    print(f"\n  Top 5 orderings (closest to 0 = best):")
    for ordering, comb, hp, nhp, rew in results[:5]:
        print(f"    {ordering}: comb={comb:.4f}  hp={hp:.4f}  nhp={nhp:.4f}")

    print(f"\n  Bottom 5 (worst):")
    for ordering, comb, hp, nhp, rew in results[-5:]:
        print(f"    {ordering}: comb={comb:.4f}  hp={hp:.4f}  nhp={nhp:.4f}")

    best = results[0]
    worst = results[-1]
    spread = best[1] - worst[1]
    print(f"\n  Best constant ordering: {best[0]} → comb={best[1]:.4f}")
    print(f"  Worst constant ordering: {worst[0]} → comb={worst[1]:.4f}")
    print(f"  Spread (best - worst): {spread:.4f}")

    return results, best


# ── Experiment 2: Greedy ordering oracle ──────────────────────────────

def greedy_ordering_oracle(scenario_id, best_cb, seed=0, max_steps=1000):
    print("\n" + "=" * 70)
    print(f"EXP 2: Greedy Ordering Oracle (per-step best of 120, CB={best_cb})")
    print("=" * 70)

    env = make_env(scenario_id, seed=seed)
    obs, _ = env.reset()

    ep_rewards, step_hp, step_nhp = [], [], []
    hp_active_total = nhp_active_total = 0
    chosen_orderings = []

    for t in range(max_steps):
        best_ord, best_r, best_info, best_env = None, -1e9, None, None

        for ordering in ALL_ORDERINGS:
            env_copy = copy.deepcopy(env)
            env_copy.set_forced_ordering(np.array(ordering, dtype=int))
            action = make_action(best_cb, ordering)
            obs_c, r_c, done_c, _, info_c = env_copy.step(action)
            if r_c > best_r:
                best_ord, best_r, best_info = ordering, r_c, info_c
                if best_env is not None:
                    try: best_env.close()
                    except: pass
                best_env = env_copy
            else:
                try: env_copy.close()
                except: pass

        try: env.close()
        except: pass
        env = best_env

        ep_rewards.append(best_r)
        hp_d, nhp_d, hp_a, nhp_a = compute_step_metrics(best_info)
        step_hp.append(hp_d)
        step_nhp.append(nhp_d)
        hp_active_total += hp_a
        nhp_active_total += nhp_a
        chosen_orderings.append(best_ord)

        if t % 100 == 0:
            print(f"    step {t:>4d}: chose {best_ord}, reward={best_r:.4f}", flush=True)

        if env is None:
            break

    try: env.close()
    except: pass

    hp_distance = sum(step_hp) / max(hp_active_total, 1)
    nhp_distance = sum(step_nhp) / max(nhp_active_total, 1)
    comb = hp_distance + nhp_distance

    from collections import Counter
    ord_dist = Counter(chosen_orderings)

    print(f"\n  Greedy ordering oracle:")
    print(f"    comb={comb:.4f}  hp={hp_distance:.4f}  nhp={nhp_distance:.4f}")
    print(f"    mean_reward={np.mean(ep_rewards):.4f}")
    print(f"    Unique orderings used: {len(ord_dist)}/{len(ALL_ORDERINGS)}")
    print(f"    Top 5 orderings: {ord_dist.most_common(5)}")

    return {
        "comb": comb, "hp_distance": hp_distance, "nhp_distance": nhp_distance,
        "mean_reward": float(np.mean(ep_rewards)),
        "n_unique": len(ord_dist),
        "ordering_dist": dict(ord_dist.most_common(10)),
    }


# ── Experiment 3: Combined greedy (CB × ordering) ────────────────────

def greedy_combined_oracle(
    scenario_id,
    seed=0,
    max_steps=1000,
    sample_orderings=20,
    sample_intras=1,
):
    """Per-step greedy over sampled CB × ordering × intra combinations."""
    n_ord = sample_orderings
    intra_candidates = sample_intra_candidates(sample_intras, seed=seed + 17)
    search_size = 11 * n_ord * len(intra_candidates)
    search_label = "CB × ordering" if len(intra_candidates) == 1 else "CB × ordering × intra"
    print("\n" + "=" * 70)
    print(f"EXP 3: Combined Greedy Oracle ({search_label} = {search_size}/step)")
    print("=" * 70)

    env_tmp = make_env(scenario_id, seed=seed)
    n_codebook = len(env_tmp.inter_quota_patterns)
    env_tmp.close()

    sampled_orderings = [
        ALL_ORDERINGS[i]
        for i in np.random.default_rng(42).choice(len(ALL_ORDERINGS), n_ord, replace=False)
    ]

    env = make_env(scenario_id, seed=seed)
    obs, _ = env.reset()

    ep_rewards, step_hp, step_nhp = [], [], []
    hp_active_total = nhp_active_total = 0
    chosen = []

    for t in range(max_steps):
        best_cb, best_ord, best_intra = -1, None, None
        best_r, best_info, best_env = -1e9, None, None

        for cb in range(n_codebook):
            for ordering in sampled_orderings:
                for intra in intra_candidates:
                    env_copy = copy.deepcopy(env)
                    env_copy.set_forced_ordering(np.array(ordering, dtype=int))
                    action = make_action(cb, ordering, intra=intra)
                    obs_c, r_c, done_c, _, info_c = env_copy.step(action)
                    if r_c > best_r:
                        best_cb = cb
                        best_ord = ordering
                        best_intra = intra
                        best_r = r_c
                        best_info = info_c
                        if best_env is not None:
                            try:
                                best_env.close()
                            except:
                                pass
                        best_env = env_copy
                    else:
                        try:
                            env_copy.close()
                        except:
                            pass

        try: env.close()
        except: pass
        env = best_env

        ep_rewards.append(best_r)
        hp_d, nhp_d, hp_a, nhp_a = compute_step_metrics(best_info)
        step_hp.append(hp_d)
        step_nhp.append(nhp_d)
        hp_active_total += hp_a
        nhp_active_total += nhp_a
        chosen.append((best_cb, best_ord, best_intra))

        if t % 100 == 0:
            print(
                f"    step {t:>4d}: CB={best_cb} ord={best_ord} "
                f"intra={best_intra} reward={best_r:.4f}",
                flush=True,
            )

        if env is None:
            break

    try: env.close()
    except: pass

    hp_distance = sum(step_hp) / max(hp_active_total, 1)
    nhp_distance = sum(step_nhp) / max(nhp_active_total, 1)
    comb = hp_distance + nhp_distance

    from collections import Counter
    cb_dist = Counter(c[0] for c in chosen)
    intra_dist = Counter(c[2] for c in chosen)

    print(f"\n  Combined greedy oracle ({search_label}):")
    print(f"    comb={comb:.4f}  hp={hp_distance:.4f}  nhp={nhp_distance:.4f}")
    print(f"    mean_reward={np.mean(ep_rewards):.4f}")
    print(f"    CB distribution: {dict(sorted(cb_dist.items()))}")
    if len(intra_candidates) > 1:
        print(f"    Top intra schedules: {intra_dist.most_common(5)}")

    return {
        "label": "Combined greedy (CB×ord)" if len(intra_candidates) == 1 else "Sampled greedy (CB×ord×intra)",
        "comb": comb, "hp_distance": hp_distance, "nhp_distance": nhp_distance,
        "mean_reward": float(np.mean(ep_rewards)),
        "cb_dist": dict(sorted(cb_dist.items())),
        "intra_dist": {str(k): v for k, v in intra_dist.most_common(10)},
        "sample_intras": len(intra_candidates),
    }


# ── Urgency ordering baseline ────────────────────────────────────────

def urgency_baseline(scenario_id, best_cb, seed=0, max_steps=1000):
    """Run with best CB + urgency heuristic ordering + RR intra (the original oracle baseline)."""
    from scripts.oracle_codebook_comparison import compute_urgency_ordering

    env = make_env(scenario_id, seed=seed)
    obs, _ = env.reset()
    step_hp, step_nhp = [], []
    hp_active_total = nhp_active_total = 0
    rewards = []

    for t in range(max_steps):
        ordering = compute_urgency_ordering(env)
        env.set_forced_ordering(ordering)
        action = make_action(best_cb, ordering)
        obs, r, done, _, info = env.step(action)
        rewards.append(r)
        hp_d, nhp_d, hp_a, nhp_a = compute_step_metrics(info)
        step_hp.append(hp_d)
        step_nhp.append(nhp_d)
        hp_active_total += hp_a
        nhp_active_total += nhp_a
        if done:
            break

    env.close()
    hp_distance = sum(step_hp) / max(hp_active_total, 1)
    nhp_distance = sum(step_nhp) / max(nhp_active_total, 1)
    comb = hp_distance + nhp_distance
    return comb, hp_distance, nhp_distance, np.mean(rewards)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--best_cb", type=int, default=0,
                        help="Best constant codebook (from previous oracle_codebook_comparison)")
    parser.add_argument("--skip_greedy", action="store_true")
    parser.add_argument("--skip_combined", action="store_true")
    parser.add_argument("--sample_orderings", type=int, default=20,
                        help="Number of orderings to sample for combined oracle")
    parser.add_argument(
        "--sample_intras",
        type=int,
        default=1,
        help="Number of intra schedules to sample for combined oracle; 1 keeps RR-only",
    )
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"Oracle Ordering Comparison — S{args.scenario}, CB={args.best_cb}, seed={args.seed}")
    print(f"{'=' * 70}")
    t0 = time.time()

    # Baseline: urgency ordering
    print("\n--- Urgency ordering baseline ---")
    urg_comb, urg_hp, urg_nhp, urg_rew = urgency_baseline(
        args.scenario, args.best_cb, args.seed, args.max_steps)
    print(f"  Urgency: comb={urg_comb:.4f}  hp={urg_hp:.4f}  nhp={urg_nhp:.4f}  reward={urg_rew:.4f}")

    # Exp 1: Constant ordering sweep
    const_results, best_const = constant_ordering_sweep(
        args.scenario, args.best_cb, args.seed, args.max_steps)

    # Exp 2: Greedy ordering oracle
    greedy_result = None
    if not args.skip_greedy:
        try:
            greedy_result = greedy_ordering_oracle(
                args.scenario, args.best_cb, args.seed, args.max_steps)
        except Exception as e:
            print(f"\n  *** Greedy ordering oracle FAILED: {e}")

    # Exp 3: Combined greedy (CB × ordering)
    combined_result = None
    if not args.skip_combined:
        try:
            combined_result = greedy_combined_oracle(
                args.scenario,
                args.seed,
                args.max_steps,
                args.sample_orderings,
                args.sample_intras,
            )
        except Exception as e:
            print(f"\n  *** Combined oracle FAILED: {e}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY — Ordering Contribution Analysis")
    print("=" * 70)
    print(f"\n{'Method':<35s} {'comb':>8s} {'hp_dist':>8s} {'nhp_dist':>8s} {'reward':>8s}")
    print("-" * 72)

    print(f"{'Urgency ordering + CB' + str(args.best_cb):<35s} "
          f"{urg_comb:>8.4f} {urg_hp:>8.4f} {urg_nhp:>8.4f} {urg_rew:>8.4f}")

    best_ord = best_const[0]
    best_comb = best_const[1]
    print(f"{'Best const ordering + CB' + str(args.best_cb):<35s} "
          f"{best_const[1]:>8.4f} {best_const[2]:>8.4f} {best_const[3]:>8.4f} {best_const[4]:>8.4f}")
    delta_const = best_const[1] - urg_comb
    pct_const = delta_const / abs(urg_comb) * 100 if urg_comb != 0 else 0
    print(f"  → Best const ordering vs urgency: {delta_const:+.4f} ({pct_const:+.1f}%)")

    if greedy_result:
        print(f"{'Greedy ordering oracle + CB' + str(args.best_cb):<35s} "
              f"{greedy_result['comb']:>8.4f} {greedy_result['hp_distance']:>8.4f} "
              f"{greedy_result['nhp_distance']:>8.4f} {greedy_result['mean_reward']:>8.4f}")
        delta_g = greedy_result['comb'] - urg_comb
        pct_g = delta_g / abs(urg_comb) * 100 if urg_comb != 0 else 0
        print(f"  → Greedy ordering oracle vs urgency: {delta_g:+.4f} ({pct_g:+.1f}%)")

    if combined_result:
        print(f"{combined_result['label']:35s} "
              f"{combined_result['comb']:>8.4f} {combined_result['hp_distance']:>8.4f} "
              f"{combined_result['nhp_distance']:>8.4f} {combined_result['mean_reward']:>8.4f}")
        delta_c = combined_result['comb'] - urg_comb
        pct_c = delta_c / abs(urg_comb) * 100 if urg_comb != 0 else 0
        print(f"  → Combined oracle vs urgency: {delta_c:+.4f} ({pct_c:+.1f}%)")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
