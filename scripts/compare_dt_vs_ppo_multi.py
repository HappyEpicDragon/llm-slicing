"""
汇总 10 个 DT 实验模型的 OOD 结果，与 ppo-multi 对比。
用法：pixi run python scripts/compare_dt_vs_ppo_multi.py
"""
import json, os
from pathlib import Path

PROJ = Path("/root/decision_transformer_slicing")
DT_EVAL = PROJ / "data/channel_generality/dt_v2_8exp/eval_ood"
PPO_MULTI = PROJ / "data/channel_generality/ppo_multi/metric_json"

EXPERIMENTS = [
    ("E1",  "D-A", "mlp"),
    ("E2",  "D-A", "slice_attn"),
    ("E3",  "D-B", "mlp"),
    ("E4",  "D-B", "slice_attn"),
    ("E5",  "D-C", "mlp"),
    ("E6",  "D-C", "slice_attn"),
    ("E7",  "D-E", "mlp"),
    ("E8",  "D-E", "slice_attn"),
    ("E9",  "D-D", "mlp"),
    ("E10", "D-D", "slice_attn"),
]

SCENARIOS = [5, 6, 7, 8, 9]
METRICS = ["hp_distance", "nhp_distance", "hp_violations", "nhp_violations"]


def load_ppo_multi():
    results = {m: [] for m in METRICS}
    for s in SCENARIOS:
        sd = PPO_MULTI / f"scenario_{s}/seed_0"
        for m in METRICS:
            f = sd / f"{m}.json"
            if f.exists():
                data = json.loads(f.read_text())
                results[m].append(data.get("mean", float("nan")))
            else:
                results[m].append(float("nan"))
    return {m: sum(v)/len(v) for m, v in results.items() if v}


def load_dt_model(exp_id):
    """从 summary.json 读取各 scenario 的均值指标。
    summary.json 键名：hp_dist_mean, nhp_dist_mean, hp_viol_mean, nhp_viol_mean
    """
    # 键名映射：metric_key → summary.json 中的键名
    KEY_MAP = {
        "hp_distance":   "hp_dist_mean",
        "nhp_distance":  "nhp_dist_mean",
        "hp_violations": "hp_viol_mean",
        "nhp_violations":"nhp_viol_mean",
    }
    vals = {m: [] for m in METRICS}
    for s in SCENARIOS:
        sf = DT_EVAL / exp_id / f"scenario_{s}/summary.json"
        if not sf.exists():
            continue
        data = json.loads(sf.read_text())
        for m in METRICS:
            v = data.get(KEY_MAP[m], None)
            if v is not None:
                vals[m].append(v)
    if not any(vals.values()):
        return None
    n = len(next(v for v in vals.values() if v))
    return {m: (sum(v)/len(v) if v else float("nan")) for m, v in vals.items()}, n


def fmt(v, ref=None, lower_better=True):
    if v != v:  # nan
        return "  N/A  "
    s = f"{v:7.4f}"
    if ref is not None and ref == ref:
        diff = v - ref
        if lower_better:
            tag = "✓" if diff < -0.001 else ("✗" if diff > 0.001 else "≈")
        else:
            tag = "✓" if diff > 0.001 else ("✗" if diff < -0.001 else "≈")
        s += tag
    return s


def main():
    ppo = load_ppo_multi()
    print(f"\n{'='*90}")
    print(f"DT 10-Exp vs PPO-Multi  (s5-s9 OOD, seed=0, mean over 5 scenarios)")
    print(f"{'='*90}")
    header = f"{'Model':<22} {'Dataset':<6} {'Encoder':<12} | {'hp_dist':>8} {'nhp_dist':>9} {'hp_viol':>8} {'nhp_viol':>9}"
    print(header)
    print("-" * 90)

    # PPO-multi 参考行
    print(f"{'ppo-multi (ref)':<22} {'--':<6} {'--':<12} | "
          f"{ppo.get('hp_distance', float('nan')):>8.4f} "
          f"{ppo.get('nhp_distance', float('nan')):>9.4f} "
          f"{ppo.get('hp_violations', float('nan')):>8.4f} "
          f"{ppo.get('nhp_violations', float('nan')):>9.4f}")
    print("-" * 90)

    for exp_id, dataset, encoder in EXPERIMENTS:
        out = load_dt_model(exp_id)
        if out is None:
            print(f"{exp_id:<22} {dataset:<6} {encoder:<12} | {'-- not ready --':>38}")
            continue
        res, n_scen = out
        label = f"{exp_id}({n_scen}/5s)"
        print(f"{label:<22} {dataset:<6} {encoder:<12} | "
              f"{fmt(res['hp_distance'],   ppo.get('hp_distance'),   True):>8} "
              f"{fmt(res['nhp_distance'],  ppo.get('nhp_distance'),  True):>9} "
              f"{fmt(res['hp_violations'], ppo.get('hp_violations'), True):>8} "
              f"{fmt(res['nhp_violations'],ppo.get('nhp_violations'),True):>9}")

    print("="*90)
    print("符号说明: ✓=优于ppo-multi  ✗=劣于ppo-multi  ≈=相当  (所有指标均为越小越好)")


if __name__ == "__main__":
    main()
