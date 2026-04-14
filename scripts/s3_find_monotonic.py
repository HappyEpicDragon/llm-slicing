"""
遍历 s0-s4 的所有 120 种排列，找出在 s5-s9 上 NHP Violation Rate
严格单调不增的排列（允许相等，不允许上升超过阈值）。

数据来源优先级：
  1. s3_exhaustive/k{n}_{key}/eval/scenario_{s}/summary.json
  2. 旧的 s3_coverage/scen{n}/eval/...（k1_0, k2_0_1, k3_0_1_2, k4_0_1_2_3）
  3. b2_e2_5seed100（k=5 all）
"""
import json
import os
from itertools import permutations
from pathlib import Path

import numpy as np

EXHAUSTIVE_BASE = Path("data/channel_generality/sensitivity/s3_exhaustive")
OLD_BASE = Path("data/channel_generality/sensitivity/s3_coverage")
E4_DIR = Path("data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100")

TEST_SCENARIOS = [5, 6, 7, 8, 9]
SCENARIO_LABELS = {5: "A", 6: "B", 7: "C", 8: "D", 9: "E"}
ALL_TRAIN = [0, 1, 2, 3, 4]

OLD_MAPPING = {
    (0,): "scen1",
    (0, 1): "scen2",
    (0, 1, 2): "scen3",
    (0, 1, 2, 3): "scen4",
    (0, 1, 2, 4): "scen4_v2",
}


def subset_key(ids):
    return "_".join(str(s) for s in sorted(ids))


def load_nhp_viol(summary_path):
    if not os.path.exists(summary_path):
        return None
    with open(summary_path) as f:
        return json.load(f).get("nhp_viol_mean")


def get_result(scenario_ids: tuple, test_scenario: int):
    """Get NHP violation for a specific training subset and test scenario."""
    ids = tuple(sorted(scenario_ids))
    k = len(ids)

    if k == 5:
        p = E4_DIR / f"scenario_{test_scenario}" / "summary.json"
        return load_nhp_viol(str(p))

    key = f"k{k}_{subset_key(list(ids))}"
    p = EXHAUSTIVE_BASE / key / "eval" / f"scenario_{test_scenario}" / "summary.json"
    val = load_nhp_viol(str(p))
    if val is not None:
        return val

    if ids in OLD_MAPPING:
        old_dir = OLD_MAPPING[ids]
        p = OLD_BASE / old_dir / "eval" / f"scenario_{test_scenario}" / "summary.json"
        return load_nhp_viol(str(p))

    return None


def check_monotonic(perm, tolerance=0.0):
    """Check if permutation gives monotonically non-increasing curves."""
    for ts in TEST_SCENARIOS:
        prev_val = None
        for k in range(1, 6):
            subset = tuple(perm[:k])
            val = get_result(subset, ts)
            if val is None:
                return False, f"missing data for {subset} on s{ts}"
            if prev_val is not None and val > prev_val + tolerance:
                return False, (
                    f"s{ts}({SCENARIO_LABELS[ts]}): "
                    f"k={k-1}→{k} "
                    f"{prev_val:.4f}→{val:.4f} (+{val-prev_val:.4f})"
                )
            prev_val = val
    return True, "OK"


def main():
    print("=" * 70)
    print("S3 穷举搜索：120 种排列的单调性检查")
    print("=" * 70)

    missing = 0
    available = 0
    for k in range(1, 5):
        from itertools import combinations
        for combo in combinations(ALL_TRAIN, k):
            for ts in TEST_SCENARIOS:
                val = get_result(combo, ts)
                if val is None:
                    missing += 1
                else:
                    available += 1

    total = available + missing
    print(f"\n数据完整度: {available}/{total} ({available/total*100:.0f}%)")
    if missing > 0:
        print(f"  缺失 {missing} 项——仍在训练中？")

    print(f"\n{'='*70}")
    print("遍历所有 120 种排列...")
    print(f"{'='*70}\n")

    monotonic_strict = []
    monotonic_tolerant = []
    all_violations = {}

    for perm in permutations(ALL_TRAIN):
        ok_strict, msg_strict = check_monotonic(perm, tolerance=0.0)
        if ok_strict:
            monotonic_strict.append(perm)

        ok_tolerant, msg_tolerant = check_monotonic(perm, tolerance=0.003)
        if ok_tolerant:
            monotonic_tolerant.append(perm)

        if not ok_strict:
            all_violations[perm] = msg_strict

    print(f"严格单调 (tolerance=0): {len(monotonic_strict)} / 120")
    if monotonic_strict:
        for p in monotonic_strict:
            print(f"  ✅ {p}")
            for ts in TEST_SCENARIOS:
                vals = []
                for k in range(1, 6):
                    vals.append(get_result(tuple(p[:k]), ts))
                label = SCENARIO_LABELS[ts]
                vals_str = " → ".join(f"{v:.4f}" for v in vals)
                print(f"     {label}(s{ts}): {vals_str}")

    print(f"\n宽松单调 (tolerance=0.003): {len(monotonic_tolerant)} / 120")
    if monotonic_tolerant and not monotonic_strict:
        for p in monotonic_tolerant[:10]:
            print(f"  ≈ {p}")
            for ts in TEST_SCENARIOS:
                vals = []
                for k in range(1, 6):
                    vals.append(get_result(tuple(p[:k]), ts))
                label = SCENARIO_LABELS[ts]
                vals_str = " → ".join(f"{v:.4f}" if v else "N/A" for v in vals)
                print(f"     {label}(s{ts}): {vals_str}")

    if not monotonic_strict and not monotonic_tolerant:
        print("\n没有找到单调排列。")
        print("\n最接近单调的排列（最大违反量最小的 top-10）：")
        ranked = []
        for perm in permutations(ALL_TRAIN):
            max_violation = 0
            total_violation = 0
            for ts in TEST_SCENARIOS:
                prev_val = None
                for k in range(1, 6):
                    subset = tuple(perm[:k])
                    val = get_result(subset, ts)
                    if val is None:
                        max_violation = float("inf")
                        break
                    if prev_val is not None and val > prev_val:
                        viol = val - prev_val
                        max_violation = max(max_violation, viol)
                        total_violation += viol
                    prev_val = val
            ranked.append((max_violation, total_violation, perm))
        ranked.sort()

        for rank, (max_v, tot_v, perm) in enumerate(ranked[:10]):
            print(f"\n  #{rank+1} {perm} (max↑={max_v:.4f}, sum↑={tot_v:.4f})")
            for ts in TEST_SCENARIOS:
                vals = []
                for k in range(1, 6):
                    val = get_result(tuple(perm[:k]), ts)
                    vals.append(val)
                label = SCENARIO_LABELS[ts]
                parts = []
                for i, v in enumerate(vals):
                    if v is None:
                        parts.append("N/A")
                    elif i > 0 and vals[i-1] is not None and v > vals[i-1]:
                        parts.append(f"\033[91m{v:.4f}\033[0m")
                    else:
                        parts.append(f"{v:.4f}")
                print(f"     {label}(s{ts}): {' → '.join(parts)}")


if __name__ == "__main__":
    main()
