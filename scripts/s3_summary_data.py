"""Dump all data used in the S2/S3 combined figure."""
import json
import os

import numpy as np

SCENARIOS = [5, 6, 7, 8, 9]
LABELS = ["A", "B", "C", "D", "E"]
E4_DIR = "data/channel_generality/dt_v2_tiny/eval_ood_100ep/b2_e2_5seed100"
S2_BASE = "data/channel_generality/sensitivity/s2_context"
EXHAUST = "data/channel_generality/sensitivity/s3_exhaustive"
OLD_S3 = "data/channel_generality/sensitivity/s3_coverage"
CQL_DIR = "data/channel_generality/cql_v3_10/cql_expert/metric_json"

PERM = (2, 1, 3, 0, 4)

OLD_MAP = {
    "k1_0": f"{OLD_S3}/scen1/eval",
    "k2_0_1": f"{OLD_S3}/scen2/eval",
    "k3_0_1_2": f"{OLD_S3}/scen3/eval",
    "k4_0_1_2_3": f"{OLD_S3}/scen4/eval",
}


def load(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get("nhp_viol_mean", float("nan"))
    return float("nan")


def subset_key(ids):
    return "_".join(str(x) for x in sorted(ids))


def find_eval(key):
    p = f"{EXHAUST}/{key}/eval"
    if os.path.isdir(p):
        return p
    return OLD_MAP.get(key, p)


# ── S2 ──
print("=" * 70)
print("(a) S2 Context Length — NHP Violation Rate")
print("=" * 70)
ctxs = [1, 5, 10, 20, 30, 50]
header = "Scen " + " ".join(f"ctx={c:>3}" for c in ctxs)
print(header)
for i, s in enumerate(SCENARIOS):
    row = []
    for ctx in ctxs:
        if ctx == 20:
            p = f"{E4_DIR}/scenario_{s}/summary.json"
        else:
            p = f"{S2_BASE}/ctx{ctx}/eval/scenario_{s}/summary.json"
        row.append(load(p))
    vals = " ".join(f"{v:>7.3f}" for v in row)
    print(f"  {LABELS[i]}   {vals}")

# ── S3 ──
print()
print("=" * 70)
print("(b) S3 Scenario Coverage — Permutation (2,1,3,0,4)")
print("=" * 70)

data = np.full((5, 5), np.nan)
subsets_desc = []
for j, n in enumerate([1, 2, 3, 4, 5]):
    subset = tuple(PERM[:n])
    ids_sorted = sorted(subset)
    subsets_desc.append(f"k={n}: {{{', '.join(str(x) for x in ids_sorted)}}}")
    for i, s in enumerate(SCENARIOS):
        if n == 5:
            p = f"{E4_DIR}/scenario_{s}/summary.json"
        else:
            key = f"k{n}_{subset_key(ids_sorted)}"
            ed = find_eval(key)
            p = f"{ed}/scenario_{s}/summary.json"
        data[i, j] = load(p)

header2 = "Scen " + " ".join(f"{'k=' + str(n):>10}" for n in [1, 2, 3, 4, 5])
print(header2)
for i in range(5):
    vals = " ".join(f"{data[i, j]:>10.4f}" for j in range(5))
    print(f"  {LABELS[i]}   {vals}")

mean = np.nanmean(data, axis=0)
std = np.nanstd(data, axis=0)
print(f" Mean  " + " ".join(f"{v:>10.4f}" for v in mean))
print(f"  Std  " + " ".join(f"{v:>10.4f}" for v in std))

print()
print("Subset composition (old scenario IDs):")
for desc in subsets_desc:
    print(f"  {desc}")

print()
print("Scenario renumbering (for paper):")
for new_id, old_id in enumerate(PERM):
    print(f"  new s{new_id} = old s{old_id}")

# CQL
cql_vals = []
for s in SCENARIOS:
    p = f"{CQL_DIR}/scenario_{s}/seed_0/nhp_violations.json"
    if os.path.exists(p):
        with open(p) as f:
            cql_vals.append(json.load(f)["mean"])
cql_mean = np.mean(cql_vals)
print(f"\nCQL reference (heatmap dashed line): mean = {cql_mean:.4f}")
print("  Per scenario: " + ", ".join(f"{LABELS[i]}={v:.4f}" for i, v in enumerate(cql_vals)))
