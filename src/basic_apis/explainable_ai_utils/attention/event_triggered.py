import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src.basic_apis.explainable_ai_utils.xai_utils import make_env, load_models, resolve_project_path


def _parse_slice_signals_from_info(info, num_slices=5):
    priorities = np.zeros(num_slices, dtype=np.float32)
    violation = np.zeros(num_slices, dtype=bool)
    fulfillment = np.zeros(num_slices, dtype=np.float32)

    for s in range(num_slices):
        priorities[s] = float(info.get(f"meta/slice_{s}_priority", 0.0))
        drifts = []
        for met in ("thr", "rel", "lat"):
            req = float(info.get(f"meta/slice_{s}_{met}_req", 0.0))
            if req <= 0:
                continue
            d = float(info.get(f"drift/slice_{s}_{met}", 0.0))
            drifts.append(d)
            v = float(info.get(f"violation/slice_{s}_{met}", 1.0 if d < 0 else 0.0))
            if v > 0.5 or d < 0:
                violation[s] = True

        fulfillment[s] = np.min(drifts) if drifts else 0.0

    # urgency proxy: 负 margin 越大 + priority 越高，紧迫度越高
    urgency = np.maximum(0.0, -fulfillment) * np.maximum(0.0, priorities)
    return violation, fulfillment, urgency


def _urgency_rank(urgency):
    # 返回按 urgency 从高到低的 slice index permutation
    idx = np.arange(len(urgency))
    # 稳定排序：先按 urgency 降序，再按 index 升序
    order = sorted(idx.tolist(), key=lambda i: (-float(urgency[i]), int(i)))
    return np.array(order, dtype=np.int32)


def _rank_from_info(info, num_slices):
    rank_pairs = []
    for s in range(num_slices):
        rank_pos = int(info.get(f"meta/slice_{s}_rank_pos", s))
        rank_pairs.append((rank_pos, s))
    rank_pairs.sort(key=lambda x: x[0])
    return np.array([s for _, s in rank_pairs], dtype=np.int32)


def _init_dt_state(dt_model, device, target_rtg):
    return {
        "history": {
            "inter": [],
            "intra": [],
            "global": [],
            "actions": [torch.zeros(len(dt_model.action_dims), dtype=torch.long, device=device)],
            "rtg": [],
            "timesteps": [],
        },
        "curr_step": 0,
        "curr_rtg": float(target_rtg),
    }


def _dt_predict_action(dt_model, obs, device, dt_state, context_len=20, rtg_scale=100000.0):
    h = dt_state["history"]
    action_dims = dt_model.action_dims
    t_inter = torch.from_numpy(obs["inter_feat"]).float().to(device)
    t_intra = torch.from_numpy(obs["intra_feat"]).float().to(device)
    t_glob = torch.from_numpy(obs["global_feat"]).float().to(device)
    h["inter"].append(t_inter)
    h["intra"].append(t_intra)
    h["global"].append(t_glob)
    rtg_input = np.sign(dt_state["curr_rtg"]) * np.log1p(np.abs(dt_state["curr_rtg"])) / rtg_scale
    h["rtg"].append(torch.tensor([rtg_input], dtype=torch.float32, device=device))
    h["timesteps"].append(torch.tensor([dt_state["curr_step"]], dtype=torch.long, device=device))

    if len(h["inter"]) > context_len:
        for k in ("inter", "intra", "global", "rtg", "timesteps", "actions"):
            h[k].pop(0)

    def _stack(lst):
        return torch.stack(lst).unsqueeze(0)

    states = {"inter": _stack(h["inter"]), "intra": _stack(h["intra"]), "global": _stack(h["global"])}
    actions_in = torch.stack(h["actions"]).unsqueeze(0)
    if actions_in.shape[1] < states["inter"].shape[1]:
        pad = torch.zeros((1, 1, len(action_dims)), dtype=torch.long, device=device)
        actions_in = torch.cat([pad, actions_in], dim=1)
    actions_in = actions_in[:, -states["inter"].shape[1]:, :]
    returns_in = _stack(h["rtg"])
    timesteps_in = _stack(h["timesteps"]).squeeze(-1)
    mask = torch.ones((1, states["inter"].shape[1]), device=device)

    with torch.no_grad():
        preds = dt_model(states, actions_in, returns_in, timesteps_in, mask)
        logits = preds[0, -1, :]

    pred = []
    start = 0
    for d in action_dims:
        end = start + d
        pred.append(int(torch.argmax(logits[start:end]).item()))
        start = end
    return np.array(pred, dtype=np.int32)


def _dt_post_step(dt_state, action, reward, device):
    dt_state["history"]["actions"].append(torch.tensor(action, dtype=torch.long, device=device))
    dt_state["curr_step"] += 1
    dt_state["curr_rtg"] -= float(reward)


def _collect_episode_trace(env, model_type, ppo_model, dt_model, device, cfg):
    obs, _ = env.reset()
    done = False
    step = 0
    max_steps = int(cfg.max_steps)

    dt_state = _init_dt_state(dt_model, device, cfg.dt_target_rtg) if model_type == "idt" else None
    rows = []
    while not done and step < max_steps:
        if model_type == "ppo":
            action, _ = ppo_model.predict(obs, deterministic=True)
        else:
            action = _dt_predict_action(dt_model, obs, device, dt_state, cfg.context_len, cfg.rtg_scale)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        slice_violation, slice_fulfillment, slice_urgency = _parse_slice_signals_from_info(info)
        slice_active_mask = np.array([int(info.get(f"meta/slice_{s}_active", 0)) for s in range(env.num_slices)], dtype=np.int32)
        rescue_trigger = int(info.get("meta/rescue_trigger", 0))
        rescue_rb_count = int(info.get("meta/rescue_rb_count", 0))
        rows.append({
            "slice_violation": slice_violation.astype(np.int32),
            "slice_resource_fraction": np.array(env.last_inter_alloc_ratio, dtype=np.float32),
            "slice_urgency": slice_urgency.astype(np.float32),
            "urgency_rank": _rank_from_info(info, env.num_slices),
            "demand_score": np.array([float(info.get(f"meta/slice_{s}_demand_score", 0.0)) for s in range(env.num_slices)], dtype=np.float32),
            "slice_fulfillment": slice_fulfillment.astype(np.float32),
            "slice_active_mask": slice_active_mask,
            "rescue_trigger": rescue_trigger,
            "rescue_rb_count": rescue_rb_count,
            "template_id": int(action[0]),
            "inter_mode_id": int(action[0]),
        })

        if model_type == "idt":
            _dt_post_step(dt_state, action, reward, device)

        obs = next_obs
        step += 1

    return rows


def _rows_to_arrays(rows):
    return {
        "slice_violation": np.stack([r["slice_violation"] for r in rows], axis=0),
        "slice_resource_fraction": np.stack([r["slice_resource_fraction"] for r in rows], axis=0),
        "slice_urgency": np.stack([r["slice_urgency"] for r in rows], axis=0),
        "urgency_rank": np.stack([r["urgency_rank"] for r in rows], axis=0),
        "demand_score": np.stack([r["demand_score"] for r in rows], axis=0),
        "slice_fulfillment": np.stack([r["slice_fulfillment"] for r in rows], axis=0),
        "slice_active_mask": np.stack([r["slice_active_mask"] for r in rows], axis=0),
        "rescue_trigger": np.array([r["rescue_trigger"] for r in rows], dtype=np.int32),
        "rescue_rb_count": np.array([r["rescue_rb_count"] for r in rows], dtype=np.int32),
        "template_id": np.array([r["template_id"] for r in rows], dtype=np.int32),
        "inter_mode_id": np.array([r["inter_mode_id"] for r in rows], dtype=np.int32),
    }


def _extract_onsets(slice_violation, include_first_if_already_violating=True):
    # crisis onset: not violated at t-1 and violated at t
    prev = slice_violation[:-1]
    curr = slice_violation[1:]
    onsets = list((np.where((prev == 0) & (curr == 1))[0] + 1).astype(int))
    if include_first_if_already_violating and len(slice_violation) > 0 and int(slice_violation[0]) == 1:
        if 0 not in onsets:
            onsets = [0] + onsets
    return np.array(onsets, dtype=np.int32)


def _aggregate_event_curves(all_traces, window, include_first_if_already_violating=True):
    # 返回 dict: resource/urgency/fulfillment 的 mean,std 与事件数
    curves_res = []
    curves_urg = []
    curves_ful = []
    severities = []

    for tr in all_traces:
        vio = tr["slice_violation"]  # [T, S]
        res = tr["slice_resource_fraction"]
        urg = tr["slice_urgency"]
        ful = tr["slice_fulfillment"]
        T, S = vio.shape
        for s in range(S):
            onsets = _extract_onsets(vio[:, s], include_first_if_already_violating=include_first_if_already_violating)
            for t0 in onsets:
                full_len = 2 * window + 1
                rr_full = np.full(full_len, np.nan, dtype=np.float32)
                uu_full = np.full(full_len, np.nan, dtype=np.float32)
                ff_full = np.full(full_len, np.nan, dtype=np.float32)

                left = max(0, t0 - window)
                right = min(T - 1, t0 + window)
                src = slice(left, right + 1)
                dst_left = window - (t0 - left)
                dst_right = dst_left + (right - left + 1)
                dst = slice(dst_left, dst_right)

                rr_full[dst] = res[src, s]
                uu_full[dst] = urg[src, s]
                ff_full[dst] = ful[src, s]

                pre_l = max(0, t0 - window)
                pre_r = max(pre_l + 1, t0)
                base_r = float(np.mean(res[pre_l:pre_r, s]))
                base_u = float(np.mean(urg[pre_l:pre_r, s]))
                base_f = float(np.mean(ful[pre_l:pre_r, s]))

                curves_res.append(rr_full - base_r)
                curves_urg.append(uu_full - base_u)
                curves_ful.append(ff_full - base_f)
                severities.append(float(abs(ful[t0, s])))

    def _pack(xs):
        if not xs:
            z = np.zeros(2 * window + 1, dtype=np.float32)
            return {"mean": z, "std": z, "n": 0}
        arr = np.stack(xs, axis=0)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
        return {
            "mean": np.nan_to_num(mean, nan=0.0),
            "std": np.nan_to_num(std, nan=0.0),
            "n": int(arr.shape[0]),
        }

    return {
        "resource": _pack(curves_res),
        "urgency": _pack(curves_urg),
        "fulfillment": _pack(curves_ful),
        "severities": np.array(severities, dtype=np.float32),
    }


def _plot_event_triggered(agg_idt, agg_ppo, out_dir, window):
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    x = np.arange(-window, window + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # (a) Resource response
    idt_m, idt_s = agg_idt["resource"]["mean"], agg_idt["resource"]["std"]
    ppo_m, ppo_s = agg_ppo["resource"]["mean"], agg_ppo["resource"]["std"]
    axes[0].plot(x, idt_m, color="#00008B", lw=2.5, label="IDT")
    axes[0].fill_between(x, idt_m - idt_s, idt_m + idt_s, color="#00008B", alpha=0.15)
    axes[0].plot(x, ppo_m, color="#d62728", lw=2.5, ls="--", label="PPO-MLP")
    axes[0].fill_between(x, ppo_m - ppo_s, ppo_m + ppo_s, color="#d62728", alpha=0.12)
    axes[0].axvline(0, color="black", lw=1.3, ls=":")
    axes[0].set_title("(a) Resource Reallocation Response", fontweight="bold")
    axes[0].set_xlabel("Relative Time (steps)")
    axes[0].set_ylabel(r"$\Delta \alpha_s$")
    axes[0].legend(loc="upper right")

    # (b) Urgency / fulfillment (IDT only)
    u_m, u_s = agg_idt["urgency"]["mean"], agg_idt["urgency"]["std"]
    f_m, f_s = agg_idt["fulfillment"]["mean"], agg_idt["fulfillment"]["std"]
    axes[1].plot(x, u_m, color="#00008B", lw=2.5, label="IDT urgency")
    axes[1].fill_between(x, u_m - u_s, u_m + u_s, color="#00008B", alpha=0.15)
    axes[1].plot(x, -f_m, color="gray", lw=2.0, ls="--", label="-fulfillment margin")
    axes[1].fill_between(x, -(f_m + f_s), -(f_m - f_s), color="gray", alpha=0.12)
    axes[1].axvline(0, color="black", lw=1.3, ls=":")
    axes[1].set_title("(b) Urgency Response (IDT)", fontweight="bold")
    axes[1].set_xlabel("Relative Time (steps)")
    axes[1].set_ylabel(r"$\Delta U_s$")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "fig10_event_triggered.png")
    pdf = os.path.join(out_dir, "fig10_event_triggered.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Event-triggered figure saved to:\n   -> {png}\n   -> {pdf}")


def _extract_episode_onsets_any_slice(slice_violation):
    # slice_violation: [T, S]
    T, S = slice_violation.shape
    onset_steps = []
    for s in range(S):
        onsets = _extract_onsets(slice_violation[:, s], include_first_if_already_violating=True)
        onset_steps.extend([int(t) for t in onsets])
    return sorted(set(onset_steps))


def _thin_events(events, min_gap=8, max_events=40):
    if not events:
        return []
    thinned = [events[0]]
    for t in events[1:]:
        if t - thinned[-1] >= min_gap:
            thinned.append(t)
    if len(thinned) > max_events:
        idx = np.linspace(0, len(thinned) - 1, num=max_events, dtype=int)
        thinned = [thinned[i] for i in idx]
    return thinned


def _rank_pos_from_perm(rank_perm):
    # rank_perm: [T,S], each row is slice ids sorted by urgency (high->low)
    T, S = rank_perm.shape
    pos = np.zeros((T, S), dtype=np.int32)
    for t in range(T):
        for p, s in enumerate(rank_perm[t]):
            pos[t, int(s)] = p + 1  # rank starts from 1
    return pos


def _rank_flip_events(rank_perm):
    # return list of (t, changed_slices)
    events = []
    for t in range(1, rank_perm.shape[0]):
        prev = rank_perm[t - 1]
        curr = rank_perm[t]
        if np.any(prev != curr):
            changed = [int(s) for s in range(rank_perm.shape[1]) if np.where(prev == s)[0][0] != np.where(curr == s)[0][0]]
            events.append((int(t), changed))
    return events


def _plot_policy_stability(
        all_idt,
        all_ppo,
        out_dir,
        seeds,
        min_gap=8,
        max_event_lines=40,
        viz_window_start=None,
        viz_window_end=None,
):
    sns.set_context("paper", font_scale=1.15)
    sns.set_style("whitegrid")

    # 用第一个 seed 画机制时序（可读性最好）
    idt_main = all_idt[0]["slice_resource_fraction"]
    ppo_main = all_ppo[0]["slice_resource_fraction"]
    idt_rank_perm = all_idt[0]["urgency_rank"]
    ppo_vio = all_ppo[0]["slice_violation"]
    T, S = idt_main.shape
    x_all = np.arange(T)

    # 可配置展示窗口（默认 50-150）
    default_start = 50 if T > 160 else 0
    default_end = min(T - 1, 150 if T > 160 else T - 1)
    if viz_window_start is not None:
        default_start = int(viz_window_start)
    if viz_window_end is not None:
        default_end = int(viz_window_end)
    w_start = max(0, min(default_start, T - 1))
    w_end = max(w_start + 1, min(default_end, T - 1))
    x = x_all[w_start:w_end + 1]

    onset_steps = _extract_episode_onsets_any_slice(ppo_vio)
    onset_steps = [t for t in onset_steps if w_start <= t <= w_end]
    onset_steps = _thin_events(onset_steps, min_gap=min_gap, max_events=max_event_lines)

    rank_flips = _rank_flip_events(idt_rank_perm)
    rank_flip_steps = [t for t, _ in rank_flips if w_start <= t <= w_end]
    rank_flip_steps = _thin_events(rank_flip_steps, min_gap=1, max_events=200)
    rank_flip_map = {t: changed for t, changed in rank_flips}
    rank_pos = _rank_pos_from_perm(idt_rank_perm)

    # spike-rank 对齐率（全时域）
    tol = 1e-9
    spikes = set((np.where(np.max(np.abs(idt_main[1:] - idt_main[:-1]), axis=1) > tol)[0] + 1).tolist())
    flips = set([t for t, _ in rank_flips])
    alignment = float(len(spikes & flips) / max(1, len(spikes)))

    # 检查是否有高度重叠的 slice（给图例说明）
    overlap_note = None
    for i in range(S):
        for j in range(i + 1, S):
            if np.max(np.abs(idt_main[:, i] - idt_main[:, j])) < 1e-6:
                overlap_note = f"Slice {i}/{j} overlap"
                break
        if overlap_note:
            break

    # 三行布局：IDT资源 / PPO资源 / IDT rank
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.2, 1.0], hspace=0.18)
    ax_idt = fig.add_subplot(gs[0])
    ax_ppo = fig.add_subplot(gs[1], sharex=ax_idt)
    ax_rank = fig.add_subplot(gs[2], sharex=ax_idt)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    for s in range(S):
        label = f"Slice {s}"
        ax_idt.plot(x, idt_main[w_start:w_end + 1, s], color=colors[s], lw=2.2, label=label)
        ax_ppo.plot(x, ppo_main[w_start:w_end + 1, s], color=colors[s], lw=2.2, label=label)
        ax_rank.step(x, rank_pos[w_start:w_end + 1, s], where="post", color=colors[s], lw=2.2, label=label)

    # rank flip 用更醒目的线，标在 (a) 和 (c)
    for t in rank_flip_steps:
        ax_idt.axvline(t, color="dimgray", lw=1.0, ls="-", alpha=0.55)
        ax_rank.axvline(t, color="dimgray", lw=1.0, ls="-", alpha=0.45)
    # PPO onset 仅标在 (b)，浅灰点线
    for t in onset_steps:
        ax_ppo.axvline(t, color="lightgray", lw=0.9, ls=":", alpha=0.75)

    # 在 rank panel 跳变点打小圆点，增强可读性
    for s in range(S):
        y = rank_pos[w_start:w_end + 1, s]
        jumps = np.where(y[1:] != y[:-1])[0] + 1
        if len(jumps) > 0:
            ax_rank.scatter(x[jumps], y[jumps], s=14, color=colors[s], zorder=4)

    ax_idt.set_title("(a) IDT Resource Allocation", fontweight="bold")
    ax_idt.set_ylabel(r"$\alpha_{s,t}$ (IDT)")
    ax_ppo.set_title("(b) PPO-MLP Resource Allocation", fontweight="bold")
    ax_ppo.set_ylabel(r"$\alpha_{s,t}$ (PPO-MLP)")
    ax_rank.set_title("(c) IDT Urgency Rank Trajectory", fontweight="bold")
    ax_rank.set_ylabel("Rank (1=highest)")
    ax_rank.set_xlabel("Simulation Steps")
    ax_idt.set_ylim(0, 0.5)
    ax_ppo.set_ylim(0, 0.5)
    ax_rank.set_ylim(S + 0.3, 0.7)  # invert to show rank-1 at top
    ax_idt.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax_idt.text(
        0.995, 0.04, f"rank flips: {len(rank_flip_steps)}",
        transform=ax_idt.transAxes, ha="right", va="bottom", fontsize=8, color="dimgray"
    )
    ax_ppo.text(
        0.995, 0.04, f"PPO onset markers: {len(onset_steps)}",
        transform=ax_ppo.transAxes, ha="right", va="bottom", fontsize=8, color="gray"
    )
    if overlap_note:
        ax_idt.text(
            0.005, 0.04, overlap_note,
            transform=ax_idt.transAxes, ha="left", va="bottom", fontsize=8, color="dimgray"
        )
    ax_rank.set_yticks(np.arange(1, S + 1))

    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "fig10_policy_stability.png")
    pdf = os.path.join(out_dir, "fig10_policy_stability.pdf")
    png2 = os.path.join(out_dir, "fig10_policy_routing.png")
    pdf2 = os.path.join(out_dir, "fig10_policy_routing.pdf")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.savefig(png2, dpi=300, bbox_inches="tight")
    plt.savefig(pdf2, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Policy-routing figure saved to:\n   -> {png}\n   -> {pdf}\n   -> {png2}\n   -> {pdf2}")

    return {
        "window_start": int(w_start),
        "window_end": int(w_end),
        "spike_rank_alignment_ratio": float(alignment),
        "rank_flip_markers_plotted": int(len(rank_flip_steps)),
        "ppo_onset_lines_plotted": int(len(onset_steps)),
    }


def _analyze_spike_sources(all_idt, tol=1e-9):
    per_seed = []
    for i, tr in enumerate(all_idt):
        res = tr["slice_resource_fraction"]          # [T,S]
        rank = tr["urgency_rank"]                    # [T,S] permutation
        active = tr["slice_active_mask"]             # [T,S]
        rescue = tr["rescue_trigger"]                # [T]

        dres = np.max(np.abs(res[1:] - res[:-1]), axis=1)
        spike_t = np.where(dres > tol)[0] + 1
        rank_t = np.where(np.any(rank[1:] != rank[:-1], axis=1))[0] + 1
        active_t = np.where(np.any(active[1:] != active[:-1], axis=1))[0] + 1
        rescue_t = np.where(rescue > 0)[0]

        spike_set = set(spike_t.tolist())
        rank_set = set(rank_t.tolist())
        active_set = set(active_t.tolist())
        rescue_set = set(rescue_t.tolist())

        n = max(1, len(spike_set))
        per_seed.append({
            "seed_index": int(i),
            "spike_count": int(len(spike_set)),
            "rank_change_count": int(len(rank_set)),
            "rescue_trigger_count": int(len(rescue_set)),
            "active_mask_change_count": int(len(active_set)),
            "spike_with_rank_change": int(len(spike_set & rank_set)),
            "spike_with_rescue": int(len(spike_set & rescue_set)),
            "spike_with_active_change": int(len(spike_set & active_set)),
            "spike_with_rank_change_ratio": float(len(spike_set & rank_set) / n),
            "spike_with_rescue_ratio": float(len(spike_set & rescue_set) / n),
            "spike_with_active_change_ratio": float(len(spike_set & active_set) / n),
        })

    def _mean(key):
        vals = [p[key] for p in per_seed]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "resource_spike_tol": float(tol),
        "per_seed": per_seed,
        "mean_spike_with_rank_change_ratio": _mean("spike_with_rank_change_ratio"),
        "mean_spike_with_rescue_ratio": _mean("spike_with_rescue_ratio"),
        "mean_spike_with_active_change_ratio": _mean("spike_with_active_change_ratio"),
    }


def execute(cfg_xai, pm, env_cfg):
    asset_dir = resolve_project_path(pm, cfg_xai.asset_dir)
    out_dir = resolve_project_path(pm, cfg_xai.output_dir)
    os.makedirs(asset_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    ppo_model, dt_model = load_models(cfg_xai, env_cfg=env_cfg)
    if ppo_model is None or dt_model is None:
        print("❌ Failed to load models. Abort event-triggered analysis.")
        return

    device = cfg_xai.device
    seeds = list(cfg_xai.seeds)
    all_idt = []
    all_ppo = []
    episode_meta = []

    for seed in seeds:
        env_idt = make_env(env_cfg, pm, cfg_xai.scenario_id, seed=seed)
        env_ppo = make_env(env_cfg, pm, cfg_xai.scenario_id, seed=seed)
        try:
            rows_idt = _collect_episode_trace(env_idt, "idt", ppo_model, dt_model, device, cfg_xai)
            rows_ppo = _collect_episode_trace(env_ppo, "ppo", ppo_model, dt_model, device, cfg_xai)
            tr_idt = _rows_to_arrays(rows_idt)
            tr_ppo = _rows_to_arrays(rows_ppo)
            all_idt.append(tr_idt)
            all_ppo.append(tr_ppo)
            episode_meta.append({"seed": int(seed), "idt_steps": int(tr_idt["slice_violation"].shape[0]), "ppo_steps": int(tr_ppo["slice_violation"].shape[0])})
            np.savez(
                os.path.join(asset_dir, f"seed_{seed}_timeseries.npz"),
                idt_slice_violation=tr_idt["slice_violation"],
                idt_slice_resource_fraction=tr_idt["slice_resource_fraction"],
                idt_slice_urgency=tr_idt["slice_urgency"],
                idt_urgency_rank=tr_idt["urgency_rank"],
                idt_demand_score=tr_idt["demand_score"],
                idt_slice_fulfillment=tr_idt["slice_fulfillment"],
                idt_slice_active_mask=tr_idt["slice_active_mask"],
                idt_rescue_trigger=tr_idt["rescue_trigger"],
                idt_rescue_rb_count=tr_idt["rescue_rb_count"],
                idt_template_id=tr_idt["template_id"],
                idt_inter_mode_id=tr_idt["inter_mode_id"],
                ppo_slice_violation=tr_ppo["slice_violation"],
                ppo_slice_resource_fraction=tr_ppo["slice_resource_fraction"],
                ppo_slice_urgency=tr_ppo["slice_urgency"],
                ppo_urgency_rank=tr_ppo["urgency_rank"],
                ppo_demand_score=tr_ppo["demand_score"],
                ppo_slice_fulfillment=tr_ppo["slice_fulfillment"],
                ppo_slice_active_mask=tr_ppo["slice_active_mask"],
                ppo_rescue_trigger=tr_ppo["rescue_trigger"],
                ppo_rescue_rb_count=tr_ppo["rescue_rb_count"],
                ppo_template_id=tr_ppo["template_id"],
                ppo_inter_mode_id=tr_ppo["inter_mode_id"],
            )
        finally:
            env_idt.close()
            env_ppo.close()

    window = int(cfg_xai.window)
    include_t0 = bool(getattr(cfg_xai, "include_first_if_already_violating", True))
    agg_idt = _aggregate_event_curves(all_idt, window, include_first_if_already_violating=include_t0)
    agg_ppo = _aggregate_event_curves(all_ppo, window, include_first_if_already_violating=include_t0)

    _plot_event_triggered(agg_idt, agg_ppo, out_dir, window)
    stability_stats = _plot_policy_stability(
        all_idt=all_idt,
        all_ppo=all_ppo,
        out_dir=out_dir,
        seeds=seeds,
        min_gap=int(getattr(cfg_xai, "event_line_min_gap", 8)),
        max_event_lines=int(getattr(cfg_xai, "event_line_max_count", 40)),
        viz_window_start=getattr(cfg_xai, "viz_window_start", None),
        viz_window_end=getattr(cfg_xai, "viz_window_end", None),
    )
    spike_source_stats = _analyze_spike_sources(
        all_idt=all_idt,
        tol=float(getattr(cfg_xai, "resource_spike_tol", 1e-9)),
    )

    summary = {
        "scenario_id": int(cfg_xai.scenario_id),
        "seeds": [int(s) for s in seeds],
        "window": int(window),
        "include_first_if_already_violating": bool(include_t0),
        "events_idt": int(agg_idt["resource"]["n"]),
        "events_ppo": int(agg_ppo["resource"]["n"]),
        "policy_stability_stats": stability_stats,
        "spike_source_stats": spike_source_stats,
        "episode_meta": episode_meta,
    }
    with open(os.path.join(asset_dir, "event_triggered_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(
        os.path.join(asset_dir, "event_triggered_curves.npz"),
        x=np.arange(-window, window + 1),
        idt_res_mean=agg_idt["resource"]["mean"],
        idt_res_std=agg_idt["resource"]["std"],
        ppo_res_mean=agg_ppo["resource"]["mean"],
        ppo_res_std=agg_ppo["resource"]["std"],
        idt_urg_mean=agg_idt["urgency"]["mean"],
        idt_urg_std=agg_idt["urgency"]["std"],
        idt_ful_mean=agg_idt["fulfillment"]["mean"],
        idt_ful_std=agg_idt["fulfillment"]["std"],
    )
    print(f"💾 Event-triggered assets saved to: {asset_dir}")

