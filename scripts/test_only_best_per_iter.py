#!/usr/bin/env python3
"""对 ReEvo 每代 proxy-best 已保存 PPO 模型跑 test_ppo_ha（每 seed 独立进程），汇总 CSV 并画演化曲线。"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"

OSC_STRIP_RE = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
OBJECTIVE_RE = re.compile(
    r"Iteration (\d+), response_id (\d+): Objective value:\s*(.+)\s*$"
)
RUN_DIR_RE = re.compile(r"versioned run dir:\s*(.+)")

CSV_FIELDS = [
    "kind",
    "iter",
    "proxy_fitness",
    "seed",
    "scenario",
    "nhp_viol_mean",
    "nhp_dist_mean",
    "hp_viol_mean",
    "hp_dist_mean",
    "reward_mean",
    "status",
]


def strip_osc(text: str) -> str:
    return OSC_STRIP_RE.sub("", text)


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_objectives_from_log(log_path: Path) -> dict[int, tuple[int, float]]:
    """按迭代分组，取有限 Objective 的最小值，返回 iter -> (response_id, fitness)."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    by_iter: dict[int, list[tuple[int, float]]] = {}
    for raw in text.splitlines():
        line = strip_osc(raw)
        m = OBJECTIVE_RE.search(line)
        if not m:
            continue
        it = int(m.group(1))
        rid = int(m.group(2))
        val_s = m.group(3).strip().lower()
        if val_s in ("inf", "infinity", "-inf", "-infinity", "nan"):
            continue
        try:
            val = float(val_s)
        except ValueError:
            continue
        if not math.isfinite(val):
            continue
        by_iter.setdefault(it, []).append((rid, val))

    best: dict[int, tuple[int, float]] = {}
    for it, pairs in by_iter.items():
        if not pairs:
            continue
        rid, v = min(pairs, key=lambda x: x[1])
        best[it] = (rid, v)
    return best


def extract_versioned_run_dir(stdout_path: Path) -> Path | None:
    """从 *_stdout.txt 中提取 versioned run dir。"""
    try:
        with open(stdout_path, encoding="utf-8", errors="replace") as f:
            for _ in range(500):
                raw = f.readline()
                if not raw:
                    break
                line = strip_osc(raw)
                rm = RUN_DIR_RE.search(line)
                if rm:
                    p = Path(rm.group(1).strip())
                    return p if p.is_absolute() else (stdout_path.parent / p).resolve()
    except OSError:
        return None
    return None


def resolve_model_zip(run_dir: Path) -> Path:
    return (run_dir / "best_model" / "best_model.zip").resolve()


def parse_summary_metrics(path: Path) -> dict[str, str]:
    """读取 metric_json/.../seed_*/summary.json。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("summary root must be object")
    nested = None
    for k, v in data.items():
        if k.startswith("scenario_") and isinstance(v, dict):
            nested = v
            break
    src: dict[str, Any] = nested if nested is not None else data

    def g(*keys: str) -> str:
        for key in keys:
            if key in src and src[key] is not None:
                return str(src[key])
        return ""

    return {
        "nhp_viol_mean": g("nhp_viol_mean", "nhp_violation_rate_mean"),
        "nhp_dist_mean": g("nhp_dist_mean"),
        "hp_viol_mean": g("hp_viol_mean", "hp_violation_rate_mean"),
        "hp_dist_mean": g("hp_dist_mean"),
        "reward_mean": g("reward_mean"),
    }


def make_test_cmd(
    model_zip: Path,
    scenario: int,
    seed: int,
    save_root: Path,
    workdir: Path,
) -> list[str]:
    sc = int(scenario)
    return [
        "pixi",
        "run",
        "python",
        "-u",
        str(MAIN_PY),
        "simulation=channel_generality/test_ppo_ha_weighted",
        f"test_ppo_ha_weighted.model_path={model_zip}",
        f"test_ppo_ha_weighted.env_updates.model_name=scenario_{sc}",
        f"test_ppo_ha_weighted.env_updates.inside.training.active_scenario_list=[{sc}]",
        f"test_ppo_ha_weighted.env_updates.inside.evaluating.active_scenario_list=[{sc}]",
        f"test_ppo_ha_weighted.env_updates.inside.testing.active_scenario_list=[{sc}]",
        f"test_ppo_ha_weighted.test_seeds=[{int(seed)}]",
        f"test_ppo_ha_weighted.save_root={save_root}",
        f"hydra.run.dir={workdir}",
    ]


@dataclass
class TestJob:
    idx: int
    kind: str
    iteration: int
    proxy_fitness: str
    seed: int
    scenario: int
    model_path: Path
    save_root: Path
    workdir: Path
    log_path: Path
    summary_json: Path
    cmd: list[str]


def build_jobs(
    search_dir: Path,
    scenario: int,
    test_seeds: Sequence[int],
    workspace: Path,
    log_dir: Path,
    baseline_model: Path | None,
    min_iteration: int,
) -> tuple[list[TestJob], list[str], dict[int, tuple[int, float]]]:
    """构造全部 test job；warnings 为校验告警。"""
    warnings: list[str] = []
    log_path = search_dir / "idt_reward-black_box.log"
    if not log_path.is_file():
        raise FileNotFoundError(f"缺少 ReEvo 日志: {log_path}")

    best_map = parse_objectives_from_log(log_path)
    iters_sorted = sorted(it for it in best_map.keys() if it >= min_iteration)
    jobs: list[TestJob] = []
    job_idx = 0
    for it in iters_sorted:
        rid, fit_v = best_map[it]
        proxy = f"{fit_v:.6g}"
        stdout_f = search_dir / f"problem_iter{it}_response{rid}.txt_stdout.txt"
        if not stdout_f.is_file():
            warnings.append(f"iter={it} 缺少 stdout 文件: {stdout_f}")
            continue
        run_dir = extract_versioned_run_dir(stdout_f)
        if run_dir is None:
            warnings.append(f"iter={it} rid={rid} 未解析到 versioned run dir")
            continue
        model_zip = resolve_model_zip(run_dir)
        if not model_zip.is_file():
            warnings.append(f"iter={it} 模型不存在: {model_zip}")

        iter_save = (workspace / f"iter{it}").resolve()
        for sd in test_seeds:
            job_idx += 1
            wd = (iter_save / f"seed{int(sd)}" / "workdir").resolve()
            summ = iter_save / "metric_json" / f"scenario_{int(scenario)}" / f"seed_{int(sd)}" / "summary.json"
            jobs.append(
                TestJob(
                    idx=job_idx,
                    kind="evolved",
                    iteration=it,
                    proxy_fitness=proxy,
                    seed=int(sd),
                    scenario=int(scenario),
                    model_path=model_zip,
                    save_root=iter_save,
                    workdir=wd,
                    log_path=(log_dir / f"evolved_iter{it}_seed{sd}.log").resolve(),
                    summary_json=summ.resolve(),
                    cmd=make_test_cmd(model_zip, scenario, int(sd), iter_save, wd),
                )
            )

    if baseline_model is not None:
        b_save = (workspace / "baseline").resolve()
        if not baseline_model.is_file():
            warnings.append(f"baseline 模型不存在: {baseline_model}")
        for sd in test_seeds:
            job_idx += 1
            wd = (b_save / f"seed{int(sd)}" / "workdir").resolve()
            summ = b_save / "metric_json" / f"scenario_{int(scenario)}" / f"seed_{int(sd)}" / "summary.json"
            jobs.append(
                TestJob(
                    idx=job_idx,
                    kind="baseline",
                    iteration=-1,
                    proxy_fitness="",
                    seed=int(sd),
                    scenario=int(scenario),
                    model_path=baseline_model.resolve(),
                    save_root=b_save,
                    workdir=wd,
                    log_path=(log_dir / f"baseline_seed{sd}.log").resolve(),
                    summary_json=summ.resolve(),
                    cmd=make_test_cmd(baseline_model, scenario, int(sd), b_save, wd),
                )
            )

    return jobs, warnings, best_map


def build_gpu_slots(gpus: Sequence[int], per_gpu: int) -> list[int]:
    slots: list[int] = []
    for g in gpus:
        slots.extend([int(g)] * int(per_gpu))
    return slots


async def _run_subprocess_logged(
    cmd: list[str],
    env: dict[str, str],
    log_path: Path,
    timeout_s: float,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab", buffering=0) as log_f:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            return proc.returncode or 0, "ok"
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return -1, "timeout"


async def run_one_test_job(
    job: TestJob,
    gpu: int,
    timeout_s: float,
    counter: list[int],
    counter_lock: asyncio.Lock,
    total: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    job.workdir.parent.mkdir(parents=True, exist_ok=True)

    async with counter_lock:
        counter[0] += 1
        n = counter[0]
    print(
        f"[{n}/{total}] iter={job.iteration} seed={job.seed} kind={job.kind} → GPU {gpu} "
        f"(save_root={job.save_root.name})",
        flush=True,
    )

    row: dict[str, str] = {
        "kind": job.kind,
        "iter": str(job.iteration),
        "proxy_fitness": job.proxy_fitness,
        "seed": str(job.seed),
        "scenario": str(job.scenario),
        "nhp_viol_mean": "",
        "nhp_dist_mean": "",
        "hp_viol_mean": "",
        "hp_dist_mean": "",
        "reward_mean": "",
        "status": "",
    }

    rc, run_st = await _run_subprocess_logged(job.cmd, env, job.log_path, float(timeout_s))
    if run_st == "timeout":
        row["status"] = "timeout"
        return row
    if rc != 0:
        row["status"] = "failed"
        return row

    try:
        m = parse_summary_metrics(job.summary_json)
        row.update(m)
        row["status"] = "ok"
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        row["status"] = "failed"
    return row


async def run_all_jobs(
    jobs: list[TestJob],
    gpu_slots: Sequence[int],
    timeout_s: float,
) -> list[dict[str, str]]:
    if not jobs:
        return []
    queue: asyncio.Queue[int] = asyncio.Queue()
    for g in gpu_slots:
        await queue.put(g)

    sem = asyncio.Semaphore(len(gpu_slots))
    counter = [0]
    counter_lock = asyncio.Lock()
    total = len(jobs)

    async def wrapped(job: TestJob) -> dict[str, str]:
        async with sem:
            gid = await queue.get()
            try:
                return await run_one_test_job(
                    job, gid, timeout_s, counter, counter_lock, total
                )
            finally:
                await queue.put(gid)

    tasks = [asyncio.create_task(wrapped(j)) for j in jobs]
    return await asyncio.gather(*tasks)


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def plot_evolution(csv_path: Path, png_path: Path, pdf_path: Path) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    rows: list[dict[str, str]] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    evolved_iters: list[int] = []
    evolved_viol_mean: list[float] = []
    evolved_viol_std: list[float] = []
    evolved_dist_mean: list[float] = []
    evolved_dist_std: list[float] = []

    baseline_viol: list[float] = []
    baseline_dist: list[float] = []

    by_iter: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r.get("kind") == "baseline":
            try:
                baseline_viol.append(float(r["nhp_viol_mean"]))
                baseline_dist.append(float(r["nhp_dist_mean"]))
            except (KeyError, ValueError):
                pass
            continue
        if r.get("kind") != "evolved":
            continue
        try:
            it = int(r["iter"])
        except (KeyError, ValueError):
            continue
        by_iter.setdefault(it, []).append(r)

    for it in sorted(by_iter.keys()):
        xs = by_iter[it]
        viol = []
        dist = []
        for x in xs:
            try:
                viol.append(float(x["nhp_viol_mean"]))
                dist.append(float(x["nhp_dist_mean"]))
            except (KeyError, ValueError):
                pass
        if not viol:
            continue
        evolved_iters.append(it)
        evolved_viol_mean.append(float(np.mean(viol)))
        evolved_viol_std.append(float(np.std(viol)))
        evolved_dist_mean.append(float(np.mean(dist)))
        evolved_dist_std.append(float(np.std(dist)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    if evolved_iters:
        iters_arr = np.array(evolved_iters, dtype=float)
        vm = np.array(evolved_viol_mean) * 100.0
        vs = np.array(evolved_viol_std) * 100.0
        ax1.errorbar(
            iters_arr,
            vm,
            yerr=vs,
            fmt="o",
            capsize=4,
            color="C0",
            label="evolved (mean ± std)",
        )

        dm = np.array(evolved_dist_mean)
        ds = np.array(evolved_dist_std)
        ax2.errorbar(iters_arr, dm, yerr=ds, fmt="o", capsize=4, color="C1")

    if baseline_viol:
        bv_m = float(np.mean(baseline_viol)) * 100.0
        bv_s = float(np.std(baseline_viol)) * 100.0
        x_min = min(evolved_iters) if evolved_iters else 0
        x_max = max(evolved_iters) if evolved_iters else 1
        ax1.axhline(bv_m, color="C3", linestyle="--", linewidth=1.5, label="baseline mean")
        ax1.fill_between([x_min, x_max], bv_m - bv_s, bv_m + bv_s, color="C3", alpha=0.2)

    if baseline_dist:
        bd_m = float(np.mean(baseline_dist))
        bd_s = float(np.std(baseline_dist))
        x_min = min(evolved_iters) if evolved_iters else 0
        x_max = max(evolved_iters) if evolved_iters else 1
        ax2.axhline(bd_m, color="C3", linestyle="--", linewidth=1.5)
        ax2.fill_between([x_min, x_max], bd_m - bd_s, bd_m + bd_s, color="C3", alpha=0.2)

    ax1.set_ylabel("NHP violation rate (%)")
    ax1.set_title("Evolution Curve: Real Metrics (test-only, 5 seeds)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2.set_xlabel("iteration")
    ax2.set_ylabel("NHP distance mean")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def parse_cli(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReEvo 每代 proxy-best PPO 模型：test_ppo_ha 单 seed 任务并行，汇总 CSV 并绘图。",
    )
    p.add_argument(
        "--search-dir",
        type=Path,
        default=None,
        help="ReEvo 输出目录（含 idt_reward-black_box.log）；--plot-only 时可省略",
    )
    p.add_argument("--scenario", type=int, default=2, help="测试场景 ID")
    p.add_argument("--test-seeds", type=str, default="0,1,2,3,4", help="逗号分隔的测试 seed 列表")
    p.add_argument("--gpus", type=str, default="0,1,2,3", help="逗号分隔的 GPU ID")
    p.add_argument("--per-gpu", type=int, default=2, help="每 GPU 并发槽位数")
    p.add_argument("--baseline-model", type=Path, default=None, help="baseline best_model.zip；不传则不测 baseline")
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/channel_generality/evolution_curve/test_only_results.csv"),
        help="汇总 CSV 路径",
    )
    p.add_argument(
        "--output-plot",
        type=Path,
        default=Path("plots/evolution_curve_test_only.png"),
        help="演化曲线 PNG 路径（同名 .pdf 一并写出）",
    )
    p.add_argument("--timeout", type=int, default=600, help="单 job 超时（秒）")
    p.add_argument(
        "--min-iteration",
        type=int,
        default=1,
        help="仅包含该迭代号及以上的最优候选（默认 1，跳过 iter0）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析日志、校验模型路径并打印任务计划，不执行 test",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="仅从已有 CSV 重画 output-plot（不跑 test）",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    test_seeds = parse_int_csv(args.test_seeds)
    gpus = parse_int_csv(args.gpus)

    csv_out = args.output_csv
    plot_out = args.output_plot
    if not csv_out.is_absolute():
        csv_out = (PROJECT_ROOT / csv_out).resolve()
    if not plot_out.is_absolute():
        plot_out = (PROJECT_ROOT / plot_out).resolve()

    baseline = args.baseline_model
    if baseline is not None and not baseline.is_absolute():
        baseline = (PROJECT_ROOT / baseline).resolve()

    if args.plot_only:
        pdf_out = plot_out.with_suffix(".pdf")
        plot_evolution(csv_out, plot_out, pdf_out)
        print(f"[plot-only] wrote {plot_out} and {pdf_out}", flush=True)
        return 0

    if args.search_dir is None:
        print("错误: 非 --plot-only 时必须提供 --search-dir", file=sys.stderr)
        return 2

    search_dir = args.search_dir.resolve()
    workspace = csv_out.parent / "test_only_workspace"
    log_dir = csv_out.parent / "test_only_logs"
    workspace.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs, warnings, best_map = build_jobs(
        search_dir,
        args.scenario,
        test_seeds,
        workspace,
        log_dir,
        baseline,
        args.min_iteration,
    )
    for w in warnings:
        print(f"[warn] {w}", flush=True)

    gpu_slots = build_gpu_slots(gpus, args.per_gpu)
    if not gpu_slots:
        print("错误: --gpus / --per-gpu 导致无可用 GPU 槽位", file=sys.stderr)
        return 1

    if args.dry_run:
        selected = sorted(i for i in best_map.keys() if i >= args.min_iteration)
        print(
            f"[dry-run] 日志中 finite 最优: {len(best_map)} 代；"
            f"min_iter>={args.min_iteration} 选中 {len(selected)} 代: {selected}",
            flush=True,
        )
        print(f"[dry-run] 计划任务数: {len(jobs)}（含 baseline={'是' if baseline else '否'}）", flush=True)
        rr = 0
        for j in jobs:
            gpu = gpu_slots[rr % len(gpu_slots)]
            rr += 1
            ok = j.model_path.is_file()
            flag = "OK" if ok else "MISSING"
            print(
                f"  job#{j.idx} iter={j.iteration} seed={j.seed} gpu→{gpu} model[{flag}]={j.model_path}",
                flush=True,
            )
        print(f"[dry-run] workspace={workspace}", flush=True)
        return 0

    # 启动前校验模型文件
    missing = [j for j in jobs if not j.model_path.is_file()]
    if missing:
        for j in missing:
            print(f"[错误] 模型缺失: {j.model_path}", file=sys.stderr)
        return 1

    rows = asyncio.run(run_all_jobs(jobs, gpu_slots, float(args.timeout)))
    write_csv(csv_out, rows)
    print(f"[done] CSV → {csv_out}", flush=True)

    pdf_out = plot_out.with_suffix(".pdf")
    plot_evolution(csv_out, plot_out, pdf_out)
    print(f"[done] plot → {plot_out} , {pdf_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
