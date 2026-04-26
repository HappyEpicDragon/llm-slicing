import csv
import json
import os
from pathlib import Path

from omegaconf import OmegaConf


PRIMARY_FIELDS = [
    "hp_viol",
    "nhp_viol",
    "total_viol",
    "mean_intent_margin",
    "violation_severity",
]

PER_QOS_FIELDS = [
    "thr_viol",
    "lat_viol",
    "rel_viol",
    "thr_margin",
    "lat_margin",
    "rel_margin",
]


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _field_value(summary, field, suffix):
    return summary.get(f"{field}_{suffix}", summary.get(field))


def _load_method(cfg, scenario_id):
    metric_root = str(cfg.metric_root)
    scenario_dir = os.path.join(metric_root, "metric_json", f"scenario_{scenario_id}")
    scenario_summary = _read_json(os.path.join(scenario_dir, "summary.json")) or {}

    seed_summaries = []
    if os.path.isdir(scenario_dir):
        for entry in sorted(os.listdir(scenario_dir)):
            if not entry.startswith("seed_"):
                continue
            seed_summary = _read_json(os.path.join(scenario_dir, entry, "summary.json"))
            if seed_summary is not None:
                seed_summaries.append(seed_summary)

    return {
        "label": str(cfg.label),
        "metric_root": metric_root,
        "model_path": str(cfg.get("model_path", "")),
        "reward_fn_path": str(cfg.get("reward_fn_path", "")),
        "scenario_summary": scenario_summary,
        "seed_summaries": seed_summaries,
    }


def _table_rows(methods, fields):
    rows = []
    for method_key, method in methods.items():
        summary = method["scenario_summary"]
        row = {
            "method": method["label"],
            "method_key": method_key,
        }
        for field in fields:
            mean = _field_value(summary, field, "mean")
            std = _field_value(summary, field, "std")
            row[f"{field}_mean"] = "" if mean is None else mean
            row[f"{field}_std"] = "" if std is None else std
        rows.append(row)
    return rows


def _markdown_table(rows, fields):
    headers = ["Method"] + [field for field in fields]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(fields)) + " |",
    ]
    for row in rows:
        cells = [row["method"]]
        for field in fields:
            mean = row.get(f"{field}_mean", "")
            std = row.get(f"{field}_std", "")
            if mean == "":
                cells.append("")
            elif std == "":
                cells.append(f"{float(mean):.6f}")
            else:
                cells.append(f"{float(mean):.6f} +/- {float(std):.6f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize_mvp(cfg):
    """Summarize scenario-0 MVP native vs agentic metric_json outputs."""
    summary_cfg = cfg.mvp_summary
    scenario_id = int(summary_cfg.scenario_id)
    output_dir = str(summary_cfg.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    methods = {}
    for method_cfg in summary_cfg.methods:
        methods[str(method_cfg.key)] = _load_method(method_cfg, scenario_id)

    primary_rows = _table_rows(methods, PRIMARY_FIELDS)
    per_qos_rows = _table_rows(methods, PER_QOS_FIELDS)

    manifest = {
        "scenario_id": scenario_id,
        "episode_split": OmegaConf.to_container(summary_cfg.episode_split, resolve=True),
        "train_seed": int(summary_cfg.train_seed),
        "test_seeds": list(summary_cfg.test_seeds),
        "training_timesteps": summary_cfg.get("training_timesteps"),
        "methods": {
            key: {
                "label": method["label"],
                "metric_root": method["metric_root"],
                "model_path": method["model_path"],
                "reward_fn_path": method["reward_fn_path"],
            }
            for key, method in methods.items()
        },
    }

    payload = {
        "manifest": manifest,
        "primary_table": primary_rows,
        "per_qos_table": per_qos_rows,
        "methods": methods,
    }

    _write_json(manifest, os.path.join(output_dir, "manifest.json"))
    _write_json(payload, os.path.join(output_dir, "mvp_summary.json"))
    _write_csv(primary_rows, os.path.join(output_dir, "primary_metrics.csv"))
    _write_csv(per_qos_rows, os.path.join(output_dir, "per_qos_metrics.csv"))

    md = [
        "# Scenario 0 MVP Summary",
        "",
        "## Primary Metrics",
        "",
        _markdown_table(primary_rows, PRIMARY_FIELDS),
        "",
        "## Per-QoS Metrics",
        "",
        _markdown_table(per_qos_rows, PER_QOS_FIELDS),
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
    ]
    md_path = Path(output_dir) / "mvp_summary.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[MVP] summary written to: {output_dir}")
    return payload
