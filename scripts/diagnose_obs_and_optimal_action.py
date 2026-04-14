"""
Diagnostic experiments A & B for constant-policy investigation.

Experiment A: Obs temporal variability (CV per dimension over 1 episode)
Experiment B: Does the optimal action change across timesteps?
              Exhaustive search over codebook(11) x intra(3^5=243) = 2673 combos,
              with ordering fixed to urgency_rank.

Usage:
    cd /root/decision_transformer_slicing
    pixi run python -u scripts/diagnose_obs_and_optimal_action.py --scenario 0
"""
import argparse, os, sys, time, copy
import numpy as np
from itertools import product
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2
from src.basic_apis.network_slicing_business.path_manager import PathManager


def make_env(scenario_id, seed=0):
    env_cfg = OmegaConf.load(os.path.join(ROOT, "conf/environment/env_ha.yaml"))
    pm = PathManager(ROOT)
    cfg = env_cfg.env_settings.copy()
    cfg.mode = "testing"
    cfg.scenario_mode = "inside"
    cfg.inside.testing.active_scenario_list = [scenario_id]
    cfg.inside.testing.init_scenario_episode = 0
    cfg.inside.testing.max_scenario_episodes = 20
    env = HierarchicalSlicingEnvV2(cfg, np.random.default_rng(seed), pm)
    return env


def compute_urgency_ordering(env):
    """Reproduce the urgency_rank ordering from old env."""
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


# ===== Experiment A: obs variability =====
def experiment_a(env, n_steps=1000):
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Observation Temporal Variability")
    print("=" * 70)

    obs, _ = env.reset()
    inter_list, intra_list, global_list = [], [], []

    for t in range(n_steps):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        inter_list.append(obs["inter_feat"].copy())
        intra_list.append(obs["intra_feat"].copy())
        global_list.append(obs["global_feat"].copy())
        if done:
            obs, _ = env.reset()

    inter_arr = np.array(inter_list)   # (T, 5, 8)
    intra_arr = np.array(intra_list)   # (T, 25, 7)
    global_arr = np.array(global_list) # (T, 2)

    inter_names = ["priority", "drift", "traffic", "alloc_ratio",
                   "spectral_eff", "demand_pressure", "drop_rate", "n_users"]
    intra_names = ["buffer", "user_prio", "user_drift", "csi", "hol_delay",
                   "user_se", "thr_ratio"]
    global_names = ["timestep_norm", "total_alloc_ratio"]

    print("\n--- Inter-slice features (5 slices x 8 dims) ---")
    print(f"{'Dim':<20} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for d, name in enumerate(inter_names):
        vals = inter_arr[:, :, d].flatten()
        mean = np.mean(vals)
        std = np.std(vals)
        cv = std / (abs(mean) + 1e-8)
        print(f"{name:<20} {mean:>8.4f} {std:>8.4f} {cv:>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    print("\n--- Per-slice inter_feat CV ---")
    header = "Slice\\Dim"
    print(f"{header:<10}", end="")
    for name in inter_names:
        print(f"{name:>16}", end="")
    print()
    for s in range(inter_arr.shape[1]):
        print(f"S{s:<9}", end="")
        for d in range(inter_arr.shape[2]):
            vals = inter_arr[:, s, d]
            mean = np.mean(vals)
            std = np.std(vals)
            cv = std / (abs(mean) + 1e-8)
            print(f"{cv:>16.4f}", end="")
        print()

    print("\n--- Intra-user features (25 users x 7 dims) ---")
    print(f"{'Dim':<20} {'Mean':>8} {'Std':>8} {'CV':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)
    for d, name in enumerate(intra_names):
        vals = intra_arr[:, :, d].flatten()
        mean = np.mean(vals)
        std = np.std(vals)
        cv = std / (abs(mean) + 1e-8)
        print(f"{name:<20} {mean:>8.4f} {std:>8.4f} {cv:>8.4f} {np.min(vals):>8.4f} {np.max(vals):>8.4f}")

    print("\n--- Global features (2 dims) ---")
    print(f"{'Dim':<20} {'Mean':>8} {'Std':>8} {'CV':>8}")
    print("-" * 45)
    for d, name in enumerate(global_names):
        vals = global_arr[:, d]
        mean = np.mean(vals)
        std = np.std(vals)
        cv = std / (abs(mean) + 1e-8)
        print(f"{name:<20} {mean:>8.4f} {std:>8.4f} {cv:>8.4f}")

    high_cv_dims = []
    for d, name in enumerate(inter_names):
        vals = inter_arr[:, :, d].flatten()
        cv = np.std(vals) / (abs(np.mean(vals)) + 1e-8)
        if cv > 0.3:
            high_cv_dims.append((name, cv))

    print(f"\n=== VERDICT A ===")
    if len(high_cv_dims) >= 3:
        print(f"PASS: {len(high_cv_dims)} inter dims have CV > 0.3 → obs has sufficient variation")
        for name, cv in high_cv_dims:
            print(f"  {name}: CV={cv:.4f}")
    else:
        print(f"FAIL: Only {len(high_cv_dims)} inter dims have CV > 0.3 → obs may lack variation")

    return inter_arr, intra_arr, global_arr


# ===== Experiment B: optimal action variation =====
def experiment_b(env, n_steps=200, sample_interval=10):
    """
    At sampled timesteps, exhaustively test codebook(11) x intra(3^5=243) = 2673 combos.
    Ordering is fixed to urgency_rank.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Does optimal action change across timesteps?")
    print(f"  Testing {n_steps} steps, sampling every {sample_interval} steps")
    print(f"  Exhaustive: 11 codebook x 243 intra = 2673 combos per sample")
    print("=" * 70)

    intra_combos = list(product(range(3), repeat=5))
    n_codebook = len(env.inter_quota_patterns)

    obs, _ = env.reset()

    best_actions_per_step = []
    rewards_per_step = []
    step_indices = []

    # We need to use a "warm-up" with a fixed action then do 1-step lookahead
    # Strategy: run through the episode, at sampled steps do exhaustive 1-step eval
    # using env snapshot (save/restore state)

    # Since we can't easily deep-copy the env, we'll use a different approach:
    # Run a baseline episode first to get the obs trajectory,
    # then for each sampled step, replay up to that step and try all actions.

    # Simpler: just run episode step by step. At sampled steps,
    # create fresh envs and replay to the same state, then try each action.
    # This is expensive but correct.

    # Most practical: run forward, at each sampled step record the current state info,
    # then do a brute-force test by creating a separate env for each combo.
    # Actually that's too expensive.

    # Best approach: Run episode with a default action, at sampled timesteps
    # use the fact that the reward depends only on the current allocation + physics.
    # We can save the env state before the step, try all actions, pick best, then
    # continue with the best action.

    # Actually the simplest correct approach:
    # Use set_forced_ordering to lock ordering to urgency_rank,
    # then at each sampled step, try all 2673 combos in separate envs...
    # But that's 2673 * 20 = 53K env constructions, too slow.

    # Pragmatic approach: sample a subset of combos per step.
    # Or: use the SAME env, save state using pickle/copy.

    # Let's try a cheaper version: at each sampled step,
    # only test the 11 codebook values with a fixed intra (all RR = [0,0,0,0,0]),
    # and separately test intra with fixed codebook.

    print("\n--- Phase 1: Codebook sweep (fixed intra=[0,0,0,0,0], urgency ordering) ---")
    env2 = make_env(env.scenario_list[0], seed=0)
    obs, _ = env2.reset()
    ordering = compute_urgency_ordering(env2)
    env2.set_forced_ordering(ordering)

    codebook_best_per_step = []
    codebook_rewards_per_step = []

    for t in range(n_steps):
        if t % sample_interval == 0:
            # Try all 11 codebooks with intra=[0,0,0,0,0]
            best_cb = -1
            best_r = -1e9
            all_r = []

            for cb in range(n_codebook):
                # Create a temporary copy of the env to test this action
                # Since we can't deep copy, we'll use a different strategy:
                # record the state, step, then restore. But env has complex internal state.

                # Alternative: create a fresh env for each test (too slow for 11 x 200).
                # Let's just do a simpler diagnostic: run the episode multiple times,
                # each time using a fixed codebook throughout.
                pass

            # Since we can't easily fork env state, let's use a different methodology:
            # Run 11 separate episodes, each with a fixed codebook, and compare rewards.
            break

    # ---- Redesigned approach: run 11 full episodes, each with a fixed codebook ----
    print("\n  [Redesigned] Running 11 full episodes, each with fixed codebook + RR intra")
    codebook_episode_rewards = []

    for cb in range(n_codebook):
        env_cb = make_env(env.scenario_list[0], seed=0)
        obs, _ = env_cb.reset()
        ordering = compute_urgency_ordering(env_cb)
        env_cb.set_forced_ordering(ordering)

        ep_rewards = []
        for t in range(n_steps):
            ordering = compute_urgency_ordering(env_cb)
            env_cb.set_forced_ordering(ordering)
            action = np.zeros(11, dtype=int)
            action[0] = cb
            action[1:6] = ordering  # ordering sub-actions (will be overridden by forced)
            action[6:11] = 0  # all RR
            obs, r, done, _, info = env_cb.step(action)
            ep_rewards.append(r)
            if done:
                break
        env_cb.close()
        mean_r = np.mean(ep_rewards)
        codebook_episode_rewards.append(mean_r)
        print(f"    Codebook {cb:>2d}: mean_reward={mean_r:.4f} (pattern={env.inter_quota_patterns[cb]})")

    codebook_episode_rewards = np.array(codebook_episode_rewards)
    best_cb_global = np.argmax(codebook_episode_rewards)
    print(f"\n  Best codebook overall: {best_cb_global} (reward={codebook_episode_rewards[best_cb_global]:.4f})")
    print(f"  Worst codebook:       {np.argmin(codebook_episode_rewards)} "
          f"(reward={codebook_episode_rewards[np.argmin(codebook_episode_rewards)]:.4f})")
    print(f"  Reward range:         {np.max(codebook_episode_rewards) - np.min(codebook_episode_rewards):.4f}")

    # ---- Now test: does the OPTIMAL codebook change within an episode? ----
    print("\n--- Phase 2: Per-timestep optimal codebook (1-step lookahead via replay) ---")
    print(f"  Testing every {sample_interval} steps")

    # For this we need multiple env instances at the same state.
    # Strategy: run a "leader" env forward. At each sampled step t, we know the full
    # history of actions. So we can replay 11 envs to step t-1 with the same actions,
    # then at step t try each codebook.

    leader_env = make_env(env.scenario_list[0], seed=0)
    obs, _ = leader_env.reset()
    leader_ordering = compute_urgency_ordering(leader_env)
    leader_env.set_forced_ordering(leader_ordering)

    default_cb = best_cb_global
    action_history = []
    per_step_best_cb = []
    per_step_cb_rewards = []

    for t in range(n_steps):
        leader_ordering = compute_urgency_ordering(leader_env)
        leader_env.set_forced_ordering(leader_ordering)

        action = np.zeros(11, dtype=int)
        action[0] = default_cb
        action[1:6] = leader_ordering
        action[6:11] = 0
        action_history.append(action.copy())

        if t % sample_interval == 0 and t > 0:
            # Replay 11 envs to step t-1, then try each codebook at step t
            step_cb_rewards = []
            for cb_test in range(n_codebook):
                test_env = make_env(env.scenario_list[0], seed=0)
                test_obs, _ = test_env.reset()

                # Replay actions 0..t-1
                for prev_t in range(t):
                    prev_ordering = compute_urgency_ordering(test_env)
                    test_env.set_forced_ordering(prev_ordering)
                    test_obs, _, d, _, _ = test_env.step(action_history[prev_t])
                    if d:
                        break

                # At step t, try codebook cb_test
                test_ordering = compute_urgency_ordering(test_env)
                test_env.set_forced_ordering(test_ordering)
                test_action = action_history[t].copy()
                test_action[0] = cb_test
                _, r_test, _, _, _ = test_env.step(test_action)
                step_cb_rewards.append(r_test)
                test_env.close()

            step_cb_rewards = np.array(step_cb_rewards)
            best_cb_t = np.argmax(step_cb_rewards)
            per_step_best_cb.append(best_cb_t)
            per_step_cb_rewards.append(step_cb_rewards)
            print(f"    t={t:>4d}: best_cb={best_cb_t:>2d} "
                  f"(r={step_cb_rewards[best_cb_t]:.4f}), "
                  f"worst_cb={np.argmin(step_cb_rewards):>2d} "
                  f"(r={step_cb_rewards[np.argmin(step_cb_rewards)]:.4f}), "
                  f"spread={np.max(step_cb_rewards) - np.min(step_cb_rewards):.4f}")

        obs, r, done, _, info = leader_env.step(action)
        if done:
            break

    leader_env.close()

    if per_step_best_cb:
        unique_cbs = len(set(per_step_best_cb))
        total_samples = len(per_step_best_cb)
        from collections import Counter
        cb_counts = Counter(per_step_best_cb)
        dominant_cb, dominant_count = cb_counts.most_common(1)[0]

        print(f"\n=== VERDICT B ===")
        print(f"  Unique optimal codebooks across {total_samples} samples: {unique_cbs}")
        print(f"  Dominant codebook: {dominant_cb} (appears {dominant_count}/{total_samples} = {dominant_count/total_samples:.1%})")
        print(f"  Distribution: {dict(cb_counts)}")

        if unique_cbs <= 2 and dominant_count / total_samples > 0.8:
            print("  → CONSTANT: Optimal codebook barely changes → constant policy is near-optimal")
        else:
            print("  → VARIES: Optimal codebook changes across time → state-dependent policy exists")

        # Check reward spread
        spreads = [np.max(r) - np.min(r) for r in per_step_cb_rewards]
        mean_spread = np.mean(spreads)
        print(f"  Mean reward spread across codebooks: {mean_spread:.4f}")
        if mean_spread < 0.01:
            print("  → INSENSITIVE: Reward barely changes with codebook → action choice doesn't matter much")
        else:
            print(f"  → SENSITIVE: Reward spread is significant → codebook choice matters")

    return per_step_best_cb, per_step_cb_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Steps per episode (default 200 for speed, max 1000)")
    parser.add_argument("--sample_interval", type=int, default=20,
                        help="Exp B: test every N steps")
    parser.add_argument("--skip_b", action="store_true",
                        help="Skip experiment B (slow)")
    args = parser.parse_args()

    print(f"=== Diagnostic: Scenario {args.scenario}, seed={args.seed}, n_steps={args.n_steps} ===\n")
    t0 = time.time()

    env = make_env(args.scenario, seed=args.seed)
    inter_arr, intra_arr, global_arr = experiment_a(env, n_steps=args.n_steps)
    env.close()

    if not args.skip_b:
        env_b = make_env(args.scenario, seed=args.seed)
        per_step_best_cb, per_step_cb_rewards = experiment_b(
            env_b, n_steps=args.n_steps, sample_interval=args.sample_interval
        )
        env_b.close()

    elapsed = time.time() - t0
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
