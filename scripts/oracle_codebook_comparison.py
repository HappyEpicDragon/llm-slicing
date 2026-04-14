"""
Oracle codebook comparison: Is constant codebook near-optimal?

Three experiments:
  1. Constant sweep: Run 11 episodes, each with a fixed codebook → best constant comb
  2. Greedy oracle: At each step, deepcopy env and try all 11 codebooks, pick best reward
  3. Hindsight bound: 11 parallel episodes (fixed codebook each), take per-step max reward

Metric: comb = hp_distance + nhp_distance (lower = better, 0 = no violation)

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/oracle_codebook_comparison.py --scenario 0
"""
import argparse, os, sys, time, copy
import numpy as np
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager

NUM_SLICES = 5


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


def compute_urgency_ordering(env):
    raw = env.last_raw_obs
    if raw is None:
        return np.arange(env.num_slices)
    slice_assoc = env.components.slices.ue_assoc
    slice_prio = raw.get("slice_priority", np.zeros(env.num_slices))
    in_bits = raw.get("pkt_incoming_bits", np.zeros(env.max_users))
    slice_traffic = (slice_assoc @ in_bits) / 1e6
    drift_raw = raw.get("intent_drift", np.zeros((env.num_slices, 5, 3)))
    slice_drift = np.min(drift_raw, axis=(1, 2))
    demand_scores = (slice_traffic * 10.0) + (slice_prio * 5.0) + ((1.0 - slice_drift) * 5.0)
    return np.argsort(demand_scores)[::-1]


def compute_step_metrics(info):
    """Per-step hp/nhp distance, aligned with metric_value.calc_intent_distance."""
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


def make_action(codebook, ordering, num_slices=5):
    action = np.zeros(1 + num_slices + num_slices, dtype=int)
    action[0] = codebook
    action[1:1 + num_slices] = ordering
    action[1 + num_slices:] = 0  # RR intra
    return action


# ── Experiment 1: Constant codebook sweep ──────────────────────────────

def run_constant_codebook(scenario_id, codebook, seed=0, max_steps=1000):
    env = make_env(scenario_id, seed=seed)
    obs, _ = env.reset()

    ep_rewards, step_hp, step_nhp = [], [], []
    hp_active_total = nhp_active_total = 0

    for t in range(max_steps):
        ordering = compute_urgency_ordering(env)
        env.set_forced_ordering(ordering)
        action = make_action(codebook, ordering)
        obs, r, done, _, info = env.step(action)
        ep_rewards.append(r)
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
    return {
        "codebook": codebook,
        "mean_reward": np.mean(ep_rewards),
        "sum_reward": np.sum(ep_rewards),
        "hp_distance": hp_distance,
        "nhp_distance": nhp_distance,
        "comb": comb,
        "n_steps": len(ep_rewards),
    }


def constant_sweep(scenario_id, seed=0, max_steps=1000):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Constant Codebook Sweep (11 codebooks)")
    print("=" * 70)

    env_tmp = make_env(scenario_id, seed=seed)
    n_codebook = len(env_tmp.inter_quota_patterns)
    patterns = [list(env_tmp.inter_quota_patterns[i]) for i in range(n_codebook)]
    env_tmp.close()

    results = []
    for cb in range(n_codebook):
        res = run_constant_codebook(scenario_id, cb, seed=seed, max_steps=max_steps)
        results.append(res)
        print(f"  CB {cb:>2d} {str(patterns[cb]):>25s}: "
              f"comb={res['comb']:.4f}  hp={res['hp_distance']:.4f}  "
              f"nhp={res['nhp_distance']:.4f}  reward={res['mean_reward']:.4f}")

    best = max(results, key=lambda x: x["comb"])   # closest to 0 = least violation
    worst = min(results, key=lambda x: x["comb"])  # most negative = worst
    print(f"\n  Best constant:  CB {best['codebook']} → comb={best['comb']:.4f}")
    print(f"  Worst constant: CB {worst['codebook']} → comb={worst['comb']:.4f}")
    return results, best


# ── Experiment 2: Greedy oracle (per-step best codebook via deepcopy) ──

def greedy_oracle(scenario_id, seed=0, max_steps=1000):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Greedy Oracle (per-step best codebook)")
    print("=" * 70)

    env = make_env(scenario_id, seed=seed)
    n_codebook = len(env.inter_quota_patterns)
    obs, _ = env.reset()

    ep_rewards, step_hp, step_nhp = [], [], []
    hp_active_total = nhp_active_total = 0
    chosen_codebooks = []

    for t in range(max_steps):
        ordering = compute_urgency_ordering(env)
        env.set_forced_ordering(ordering)

        best_cb, best_r, best_info, best_obs, best_env = -1, -1e9, None, None, None

        for cb in range(n_codebook):
            env_copy = copy.deepcopy(env)
            action = make_action(cb, ordering)
            obs_c, r_c, done_c, _, info_c = env_copy.step(action)
            if r_c > best_r:
                best_cb, best_r, best_info, best_obs = cb, r_c, info_c, obs_c
                if best_env is not None:
                    try:
                        best_env.close()
                    except Exception:
                        pass
                best_env = env_copy
            else:
                try:
                    env_copy.close()
                except Exception:
                    pass

        try:
            env.close()
        except Exception:
            pass
        env = best_env

        ep_rewards.append(best_r)
        hp_d, nhp_d, hp_a, nhp_a = compute_step_metrics(best_info)
        step_hp.append(hp_d)
        step_nhp.append(nhp_d)
        hp_active_total += hp_a
        nhp_active_total += nhp_a
        chosen_codebooks.append(best_cb)

        if t % 100 == 0:
            print(f"    step {t:>4d}: chose CB {best_cb}, reward={best_r:.4f}")

        if env is None:
            break

    try:
        env.close()
    except Exception:
        pass

    hp_distance = sum(step_hp) / max(hp_active_total, 1)
    nhp_distance = sum(step_nhp) / max(nhp_active_total, 1)
    comb = hp_distance + nhp_distance

    from collections import Counter
    cb_dist = Counter(chosen_codebooks)

    print(f"\n  Greedy oracle result:")
    print(f"    comb={comb:.4f}  hp={hp_distance:.4f}  nhp={nhp_distance:.4f}")
    print(f"    mean_reward={np.mean(ep_rewards):.4f}")
    print(f"    Codebook distribution: {dict(sorted(cb_dist.items()))}")
    print(f"    Unique codebooks used: {len(cb_dist)}")

    return {
        "mean_reward": np.mean(ep_rewards),
        "sum_reward": np.sum(ep_rewards),
        "hp_distance": hp_distance,
        "nhp_distance": nhp_distance,
        "comb": comb,
        "codebook_dist": dict(sorted(cb_dist.items())),
        "n_unique_codebooks": len(cb_dist),
    }


# ── Experiment 3: Hindsight bound ────────────────────────────────────

def hindsight_bound(scenario_id, seed=0, max_steps=1000):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Hindsight Bound (per-step max across parallel runs)")
    print("=" * 70)

    env_tmp = make_env(scenario_id, seed=seed)
    n_codebook = len(env_tmp.inter_quota_patterns)
    env_tmp.close()

    all_envs = [make_env(scenario_id, seed=seed) for _ in range(n_codebook)]
    for e in all_envs:
        e.reset()

    per_step_rewards = np.zeros((max_steps, n_codebook))
    per_step_hp = np.zeros((max_steps, n_codebook))
    per_step_nhp = np.zeros((max_steps, n_codebook))
    per_step_hp_active = np.zeros((max_steps, n_codebook))
    per_step_nhp_active = np.zeros((max_steps, n_codebook))

    actual_steps = max_steps
    for t in range(max_steps):
        for cb in range(n_codebook):
            ordering = compute_urgency_ordering(all_envs[cb])
            all_envs[cb].set_forced_ordering(ordering)
            action = make_action(cb, ordering)
            _, r, done, _, info = all_envs[cb].step(action)
            per_step_rewards[t, cb] = r
            hp_d, nhp_d, hp_a, nhp_a = compute_step_metrics(info)
            per_step_hp[t, cb] = hp_d
            per_step_nhp[t, cb] = nhp_d
            per_step_hp_active[t, cb] = hp_a
            per_step_nhp_active[t, cb] = nhp_a
            if done:
                actual_steps = min(actual_steps, t + 1)

    for e in all_envs:
        e.close()

    per_step_rewards = per_step_rewards[:actual_steps]
    per_step_hp = per_step_hp[:actual_steps]
    per_step_nhp = per_step_nhp[:actual_steps]
    per_step_hp_active = per_step_hp_active[:actual_steps]
    per_step_nhp_active = per_step_nhp_active[:actual_steps]

    best_cb_per_step = np.argmax(per_step_rewards, axis=1)
    hindsight_rewards = per_step_rewards[np.arange(actual_steps), best_cb_per_step]
    hindsight_hp = per_step_hp[np.arange(actual_steps), best_cb_per_step]
    hindsight_nhp = per_step_nhp[np.arange(actual_steps), best_cb_per_step]
    hindsight_hp_a = per_step_hp_active[np.arange(actual_steps), best_cb_per_step]
    hindsight_nhp_a = per_step_nhp_active[np.arange(actual_steps), best_cb_per_step]

    hp_distance = np.sum(hindsight_hp) / max(np.sum(hindsight_hp_a), 1)
    nhp_distance = np.sum(hindsight_nhp) / max(np.sum(hindsight_nhp_a), 1)
    comb = hp_distance + nhp_distance

    from collections import Counter
    cb_dist = Counter(best_cb_per_step.tolist())

    print(f"\n  Hindsight bound (per-step max, upper bound on oracle):")
    print(f"    comb={comb:.4f}  hp={hp_distance:.4f}  nhp={nhp_distance:.4f}")
    print(f"    mean_reward={np.mean(hindsight_rewards):.4f}")
    print(f"    Codebook distribution: {dict(sorted(cb_dist.items()))}")

    return {
        "mean_reward": float(np.mean(hindsight_rewards)),
        "hp_distance": hp_distance,
        "nhp_distance": nhp_distance,
        "comb": comb,
        "codebook_dist": dict(sorted(cb_dist.items())),
    }


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--skip_greedy", action="store_true",
                        help="Skip greedy oracle (slow due to deepcopy)")
    args = parser.parse_args()

    print(f"{'=' * 70}")
    print(f"Oracle Codebook Comparison — Scenario {args.scenario}, seed={args.seed}")
    print(f"{'=' * 70}")
    t0 = time.time()

    # Exp 1: constant sweep
    const_results, best_const = constant_sweep(
        args.scenario, seed=args.seed, max_steps=args.max_steps)

    # Exp 2: greedy oracle
    greedy_result = None
    if not args.skip_greedy:
        try:
            greedy_result = greedy_oracle(
                args.scenario, seed=args.seed, max_steps=args.max_steps)
        except Exception as e:
            print(f"\n  *** Greedy oracle FAILED (deepcopy issue): {e}")
            print(f"  *** Falling back to hindsight bound only")

    # Exp 3: hindsight bound
    hind_result = hindsight_bound(
        args.scenario, seed=args.seed, max_steps=args.max_steps)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY — Constant vs Oracle Codebook Selection")
    print("=" * 70)
    print(f"\n{'Method':<25s} {'comb':>8s} {'hp_dist':>8s} {'nhp_dist':>8s} {'reward':>8s}")
    print("-" * 60)
    print(f"{'Best constant (CB ' + str(best_const['codebook']) + ')':<25s} "
          f"{best_const['comb']:>8.4f} {best_const['hp_distance']:>8.4f} "
          f"{best_const['nhp_distance']:>8.4f} {best_const['mean_reward']:>8.4f}")

    if greedy_result:
        print(f"{'Greedy oracle':<25s} "
              f"{greedy_result['comb']:>8.4f} {greedy_result['hp_distance']:>8.4f} "
              f"{greedy_result['nhp_distance']:>8.4f} {greedy_result['mean_reward']:>8.4f}")
        delta_greedy = greedy_result['comb'] - best_const['comb']  # positive = oracle better
        pct = delta_greedy / abs(best_const['comb']) * 100 if best_const['comb'] != 0 else 0
        print(f"  → Greedy improvement over constant: {delta_greedy:+.4f} ({pct:+.1f}%)")

    print(f"{'Hindsight bound':<25s} "
          f"{hind_result['comb']:>8.4f} {hind_result['hp_distance']:>8.4f} "
          f"{hind_result['nhp_distance']:>8.4f} {hind_result['mean_reward']:>8.4f}")
    delta_hind = hind_result['comb'] - best_const['comb']  # positive = oracle better
    pct_h = delta_hind / abs(best_const['comb']) * 100 if best_const['comb'] != 0 else 0
    print(f"  → Hindsight improvement over constant: {delta_hind:+.4f} ({pct_h:+.1f}%)")

    print(f"\n{'=' * 70}")
    if greedy_result:
        ref_comb = greedy_result['comb']
        ref_name = "greedy oracle"
    else:
        ref_comb = hind_result['comb']
        ref_name = "hindsight bound"

    # comb is negative; closer to 0 = better.  oracle should be >= best_const.
    delta = ref_comb - best_const['comb']          # positive if oracle is better
    pct_final = abs(delta / best_const['comb']) * 100 if best_const['comb'] != 0 else 0

    if pct_final < 5.0:
        print(f"VERDICT: Constant codebook is NEAR-OPTIMAL (<5% gap to {ref_name})")
        print(f"  → Problem is NOT in codebook switching; focus on action space design")
    else:
        print(f"VERDICT: Oracle is SIGNIFICANTLY BETTER ({pct_final:.1f}% gap)")
        print(f"  → State-dependent codebook matters; fix entropy collapse (ent_coef↑)")

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
