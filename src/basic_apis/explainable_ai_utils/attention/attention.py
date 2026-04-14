import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.basic_apis.explainable_ai_utils.xai_utils import make_env, load_models, resolve_project_path


def _infer_num_slices_from_info(info, default=5):
    if not isinstance(info, dict):
        return default
    slice_ids = set()
    for k in info.keys():
        if k.startswith("meta/slice_"):
            parts = k.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                slice_ids.add(int(parts[1]))
    return (max(slice_ids) + 1) if slice_ids else default


def _clone_obs(obs):
    """深拷贝 observation，避免后续 env.step 覆盖引用。"""
    return {k: np.array(v, copy=True) for k, v in obs.items()}


def _extract_violation_event(info):
    """
    从 env.step 返回的 info 中提取最严重的 HP violation 事件。
    以 |negative drift| 作为严重度。
    """
    if not isinstance(info, dict):
        return None

    best = None
    for s_idx in range(_infer_num_slices_from_info(info)):
        if info.get(f"meta/slice_{s_idx}_priority", 0) <= 0:
            continue
        for met in ("thr", "rel", "lat"):
            if info.get(f"meta/slice_{s_idx}_{met}_req", 0) <= 0:
                continue
            drift = float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
            if drift >= 0.0:
                continue

            cand = {
                "slice_idx": int(s_idx),
                "metric": met,
                "drift": drift,
                "severity": abs(drift),
            }
            if best is None or cand["severity"] > best["severity"]:
                best = cand
    return best


def _count_hp_violations(info):
    if not isinstance(info, dict):
        return 0, 0.0
    cnt = 0
    sev = 0.0
    for s_idx in range(_infer_num_slices_from_info(info)):
        if info.get(f"meta/slice_{s_idx}_priority", 0) <= 0:
            continue
        for met in ("thr", "rel", "lat"):
            if info.get(f"meta/slice_{s_idx}_{met}_req", 0) <= 0:
                continue
            drift = float(info.get(f"drift/slice_{s_idx}_{met}", 0.0))
            if drift < 0.0:
                cnt += 1
                sev += abs(drift)
    return cnt, sev


def _pick_user_in_slice(obs, slice_idx, slice_assoc):
    """
    在指定 slice 内选择当前 buffer 最大的用户，用于画蓝框。
    """
    users = np.where(slice_assoc[slice_idx] == 1)[0]
    if len(users) == 0:
        return 0, 0.0
    buffers = obs["intra_feat"][users, 0]
    local_idx = int(np.argmax(buffers))
    user_idx = int(users[local_idx])
    return user_idx, float(buffers[local_idx])


def hunt_for_crisis(env, dt_model, cfg_xai):
    """
    运行环境，寻找高风险时刻。
    使用随机动作推进环境（专注于找危机，不需要精确 DT 推理）。
    """
    device = cfg_xai.device
    print(f"🕵️ Hunting for crisis in Scenario {cfg_xai.scenario_id} (signal=violation/drift first)...")

    obs, _ = env.reset()

    # DT Context Init
    context_len = 20
    action_dims = dt_model.action_dims
    dummy_action = torch.zeros(len(action_dims), dtype=torch.long, device=device)

    history = {
        "inter": [], "intra": [], "global": [],
        "actions": [dummy_action],
        "rtg": [], "timesteps": []
    }

    curr_rtg = -20000.0
    rtg_scale = float(getattr(cfg_xai, "rtg_scale", 100000.0))
    curr_step = 0
    max_steps = int(getattr(cfg_xai, "search_max_steps", 200))
    fallback_to_peak = bool(getattr(cfg_xai, "fallback_to_peak", True))
    best_obs = None  # buffer fallback
    best_user = None
    best_slice = None
    best_buf_val = -1.0
    best_step = -1
    best_violation = None
    violation_frames = []
    return_first_violation = bool(getattr(cfg_xai, "return_first_violation", False))
    scan_full_episode = bool(getattr(cfg_xai, "scan_full_episode", True))
    min_step_for_violation = int(getattr(cfg_xai, "min_step_for_violation", 5))
    episodes_seen = 0

    for step in range(max_steps):
        # --- 1. 记录 buffer 峰值（兜底） ---
        buffers = obs['intra_feat'][:, 0]
        max_buf_val = np.max(buffers)
        max_buf_user = np.argmax(buffers)
        slice_assoc = env.components.slices.ue_assoc  # [5, 25]
        slice_idx = np.where(slice_assoc[:, max_buf_user] == 1)[0][0]

        if max_buf_val > best_buf_val:
            best_buf_val = float(max_buf_val)
            best_user = int(max_buf_user)
            best_slice = int(slice_idx)
            best_obs = _clone_obs(obs)
            best_step = int(step)

        # --- 2. 推进环境 ---
        t_inter = torch.from_numpy(obs['inter_feat']).float().to(device)
        t_intra = torch.from_numpy(obs['intra_feat']).float().to(device)
        t_glob = torch.from_numpy(obs['global_feat']).float().to(device)

        history["inter"].append(t_inter)
        history["intra"].append(t_intra)
        history["global"].append(t_glob)

        rtg_val = np.sign(curr_rtg) * np.log1p(np.abs(curr_rtg)) / rtg_scale
        history["rtg"].append(torch.tensor([rtg_val], dtype=torch.float32, device=device))
        history["timesteps"].append(torch.tensor([curr_step], dtype=torch.long, device=device))

        if len(history["inter"]) > context_len:
            for k in ["inter", "intra", "global", "rtg", "timesteps", "actions"]:
                history[k].pop(0)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        curr_rtg -= reward
        curr_step += 1

        history["actions"].append(torch.tensor(action, dtype=torch.long, device=device))

        # --- 3. 危机判定：优先 violation/drift ---
        event = _extract_violation_event(info)
        if event is not None:
            viol_slice = event["slice_idx"]
            viol_user, viol_user_buf = _pick_user_in_slice(next_obs, viol_slice, slice_assoc)
            event_payload = {
                "obs": _clone_obs(next_obs),
                "slice_idx": int(viol_slice),
                "user_idx": int(viol_user),
                "step": int(step),
                "user_buffer": float(viol_user_buf),
                "violation_metric": event["metric"],
                "violation_drift": float(event["drift"]),
                "violation_severity": float(event["severity"]),
            }
            violation_frames.append(event_payload)
            if best_violation is None or event_payload["violation_severity"] > best_violation["violation_severity"]:
                best_violation = event_payload

            if return_first_violation and (step >= min_step_for_violation) and (not scan_full_episode):
                print(f"🚨 VIOLATION FOUND at Step {step} | slice={viol_slice}, metric={event['metric']}, drift={event['drift']:.4f}")
                return event_payload["obs"], event_payload["slice_idx"], event_payload["user_idx"], {
                    "selection_mode": "violation_hit",
                    "selected_step": event_payload["step"],
                    "selected_buffer": event_payload["user_buffer"],
                    "violation_metric": event_payload["violation_metric"],
                    "violation_drift": event_payload["violation_drift"],
                    "violation_severity": event_payload["violation_severity"],
                }, violation_frames

        obs = next_obs
        if done:
            episodes_seen += 1
            if scan_full_episode:
                break
            obs, _ = env.reset()
            curr_step = 0

    if best_violation is not None:
        print(f"🚨 Using strongest violation snapshot after {episodes_seen if episodes_seen > 0 else 1} episode: step={best_violation['step']}, "
              f"slice={best_violation['slice_idx']}, metric={best_violation['violation_metric']}, "
              f"drift={best_violation['violation_drift']:.4f}")
        return best_violation["obs"], best_violation["slice_idx"], best_violation["user_idx"], {
            "selection_mode": "violation_peak_episode",
            "selected_step": best_violation["step"],
            "selected_buffer": best_violation["user_buffer"],
            "violation_metric": best_violation["violation_metric"],
            "violation_drift": best_violation["violation_drift"],
            "violation_severity": best_violation["violation_severity"],
            "episodes_scanned": int(episodes_seen if episodes_seen > 0 else 1),
        }, violation_frames

    print("⚠️ Max steps reached without finding violation-defined crisis.")
    if fallback_to_peak and best_obs is not None:
        print(f"🔁 Fallback to peak buffer snapshot: step={best_step}, user={best_user}, buffer={best_buf_val:.2f}")
        return best_obs, best_slice, best_user, {
            "selection_mode": "peak_buffer_fallback",
            "selected_step": best_step,
            "selected_buffer": best_buf_val,
        }, violation_frames
    return None, None, None, {
        "selection_mode": "not_found",
        "selected_step": -1,
        "selected_buffer": -1.0,
    }, violation_frames


def extract_attention_maps(ppo_model, dt_model, obs, device):
    """
    分别从 PPO 和 DT 中提取 Attention Weights
    """
    # === PPO ===
    ppo_obs_tensor, _ = ppo_model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        # ppo_attn shape: [Batch=1, Num_Heads=4, Num_Slices=5, Num_Users=25]
        _, ppo_attn = ppo_model.policy.features_extractor(ppo_obs_tensor, return_attn=True)

    # === DT ===
    t_inter = torch.from_numpy(obs['inter_feat']).float().to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, 5, 4]
    t_intra = torch.from_numpy(obs['intra_feat']).float().to(device).unsqueeze(0).unsqueeze(0)
    t_glob = torch.from_numpy(obs['global_feat']).float().to(device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        _, dt_attn = dt_model.state_encoder(t_inter, t_intra, t_glob, return_attn=True)

    return ppo_attn.squeeze().cpu().numpy(), dt_attn.squeeze().cpu().numpy()


def _to_2d_attn(attn):
    return np.mean(attn, axis=0) if attn.ndim == 3 else attn


def _dt_embedding_from_obs(dt_model, obs, device):
    t_inter = torch.from_numpy(obs['inter_feat']).float().to(device).unsqueeze(0).unsqueeze(0)
    t_intra = torch.from_numpy(obs['intra_feat']).float().to(device).unsqueeze(0).unsqueeze(0)
    t_glob = torch.from_numpy(obs['global_feat']).float().to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        emb = dt_model.state_encoder(t_inter, t_intra, t_glob).squeeze(0).squeeze(0)
    return emb.cpu().numpy()


def _dt_logits_from_obs(dt_model, obs, device):
    action_dims = dt_model.action_dims
    states = {
        "inter": torch.from_numpy(obs['inter_feat']).float().to(device).unsqueeze(0).unsqueeze(0),
        "intra": torch.from_numpy(obs['intra_feat']).float().to(device).unsqueeze(0).unsqueeze(0),
        "global": torch.from_numpy(obs['global_feat']).float().to(device).unsqueeze(0).unsqueeze(0),
    }
    actions = torch.zeros((1, 1, len(action_dims)), dtype=torch.long, device=device)
    returns = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
    timesteps = torch.zeros((1, 1), dtype=torch.long, device=device)
    mask = torch.ones((1, 1), dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = dt_model(
            states=states,
            actions=actions,
            returns=returns,
            timesteps=timesteps,
            attention_mask=mask
        )[0, -1, :]
    return logits.cpu().numpy()


def _decode_multidiscrete_action(logits, action_dims):
    acts = []
    start = 0
    for d in action_dims:
        end = start + d
        acts.append(int(np.argmax(logits[start:end])))
        start = end
    return acts


def _multidiscrete_kl(base_logits, cf_logits, action_dims):
    """
    计算 MultiDiscrete 动作分布的平均 KL(base || cf)。
    """
    kls = []
    start = 0
    for d in action_dims:
        end = start + d
        b = base_logits[start:end]
        c = cf_logits[start:end]
        b = b - np.max(b)
        c = c - np.max(c)
        pb = np.exp(b) / (np.sum(np.exp(b)) + 1e-12)
        pc = np.exp(c) / (np.sum(np.exp(c)) + 1e-12)
        kl = np.sum(pb * (np.log(pb + 1e-12) - np.log(pc + 1e-12)))
        kls.append(float(kl))
        start = end
    return float(np.mean(kls))


def _cosine_similarity(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _replace_user_with_slice_mean(obs, user_idx, slice_assoc):
    obs_new = _clone_obs(obs)
    s_idx = int(np.where(slice_assoc[:, user_idx] == 1)[0][0])
    users = np.where(slice_assoc[s_idx] == 1)[0]
    obs_new["intra_feat"][user_idx] = np.mean(obs_new["intra_feat"][users], axis=0)
    return obs_new


def _framewise_counterfactual_panel(dt_model, device, slice_assoc, frames, users):
    """
    在多帧 violation 样本上做反事实，统计 logits KL 和动作变化率。
    """
    panel = []
    for user_idx in users:
        recs = []
        for fr in frames:
            obs = fr["obs"]
            base_logits = _dt_logits_from_obs(dt_model, obs, device)
            base_action = _decode_multidiscrete_action(base_logits, dt_model.action_dims)

            obs_cf = _replace_user_with_slice_mean(obs, user_idx, slice_assoc)
            cf_logits = _dt_logits_from_obs(dt_model, obs_cf, device)
            cf_action = _decode_multidiscrete_action(cf_logits, dt_model.action_dims)

            recs.append({
                "step": int(fr["step"]),
                "slice": int(fr["slice_idx"]),
                "action_changed": bool(np.any(np.array(base_action) != np.array(cf_action))),
                "kl_mean": _multidiscrete_kl(base_logits, cf_logits, dt_model.action_dims),
                "logits_l1": float(np.mean(np.abs(cf_logits - base_logits))),
            })

        if recs:
            panel.append({
                "user": int(user_idx),
                "frames": int(len(recs)),
                "action_change_rate": float(np.mean([1.0 if r["action_changed"] else 0.0 for r in recs])),
                "kl_mean": float(np.mean([r["kl_mean"] for r in recs])),
                "kl_std": float(np.std([r["kl_mean"] for r in recs])),
                "logits_l1_mean": float(np.mean([r["logits_l1"] for r in recs])),
                "records": recs,
            })
    return panel


def _dt_action_from_history_obs(dt_model, obs, history, curr_rtg, curr_step, device, context_len=20, rtg_scale=100000.0):
    action_dims = dt_model.action_dims
    t_inter = torch.from_numpy(obs['inter_feat']).float().to(device)
    t_intra = torch.from_numpy(obs['intra_feat']).float().to(device)
    t_glob = torch.from_numpy(obs['global_feat']).float().to(device)

    history["inter"].append(t_inter)
    history["intra"].append(t_intra)
    history["global"].append(t_glob)
    rtg_input = np.sign(curr_rtg) * np.log1p(np.abs(curr_rtg)) / rtg_scale
    history["rtg"].append(torch.tensor([rtg_input], dtype=torch.float32, device=device))
    history["timesteps"].append(torch.tensor([curr_step], dtype=torch.long, device=device))

    if len(history["inter"]) > context_len:
        for k in ["inter", "intra", "global", "rtg", "timesteps", "actions"]:
            history[k].pop(0)

    def stack_helper(lst):
        return torch.stack(lst).unsqueeze(0)

    states = {
        'inter': stack_helper(history['inter']),
        'intra': stack_helper(history['intra']),
        'global': stack_helper(history['global'])
    }
    actions_in = torch.stack(history['actions']).unsqueeze(0)
    if actions_in.shape[1] < states['inter'].shape[1]:
        actions_in = torch.cat(
            [torch.zeros((1, 1, len(action_dims)), dtype=torch.long, device=device), actions_in], dim=1
        )
    actions_in = actions_in[:, -states['inter'].shape[1]:, :]
    returns_in = stack_helper(history['rtg'])
    timesteps_in = stack_helper(history['timesteps']).squeeze(-1)
    mask = torch.ones((1, states['inter'].shape[1]), device=device)

    with torch.no_grad():
        preds = dt_model(states, actions_in, returns_in, timesteps_in, mask)
        logits = preds[0, -1, :].detach().cpu().numpy()
    action = _decode_multidiscrete_action(logits, action_dims)
    return np.array(action, dtype=np.int32), logits


def _run_dt_rollout_with_intervention(env, dt_model, intervention_user=None, max_steps=200, target_rtg=-20000.0):
    obs, _ = env.reset()
    total_reward = 0.0
    violations = 0
    violation_severity = 0.0
    action_change_proxy = 0
    kl_acc = []
    steps = 0
    slice_assoc = env.components.slices.ue_assoc
    device = next(dt_model.parameters()).device
    action_dims = dt_model.action_dims
    curr_rtg = float(target_rtg)
    curr_step = 0

    def _init_history():
        return {
            "inter": [], "intra": [], "global": [],
            "actions": [torch.zeros(len(action_dims), dtype=torch.long, device=device)],
            "rtg": [], "timesteps": []
        }

    history_base = _init_history()
    history_cf = _init_history()
    for _ in range(max_steps):
        base_action, base_logits = _dt_action_from_history_obs(
            dt_model, obs, history_base, curr_rtg, curr_step, device
        )

        if intervention_user is None:
            action = base_action
        else:
            obs_cf = _replace_user_with_slice_mean(obs, intervention_user, slice_assoc)
            cf_action, cf_logits = _dt_action_from_history_obs(
                dt_model, obs_cf, history_cf, curr_rtg, curr_step, device
            )
            action = cf_action
            kl_acc.append(_multidiscrete_kl(base_logits, cf_logits, action_dims))
            if np.any(base_action != cf_action):
                action_change_proxy += 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += float(reward)
        vc, vs = _count_hp_violations(info)
        violations += vc
        violation_severity += vs
        steps += 1
        curr_rtg -= float(reward)
        curr_step += 1

        # 对 baseline 也要维护 actions 历史（用于下一步因果链）
        history_base["actions"].append(torch.tensor(base_action, dtype=torch.long, device=device))
        if intervention_user is not None:
            history_cf["actions"].append(torch.tensor(action, dtype=torch.long, device=device))

        if done:
            break

    return {
        "steps": int(steps),
        "return": float(total_reward),
        "hp_violation_count": int(violations),
        "hp_violation_severity": float(violation_severity),
        "action_change_proxy_rate": float(action_change_proxy / max(1, steps)),
        "kl_mean": float(np.mean(kl_acc)) if kl_acc else 0.0,
    }


def run_rollout_impact(env_cfg, pm, cfg_xai, dt_model, anchor_user, control_user):
    seeds = list(range(int(getattr(cfg_xai, "anchor_rollout_seeds", 3))))
    max_steps = int(getattr(cfg_xai, "anchor_rollout_steps", 200))
    target_rtg = float(getattr(cfg_xai, "anchor_rollout_target_rtg", -20000.0))
    groups = {
        "baseline": [],
        "anchor_cf": [],
        "control_cf": [],
    }
    for seed in seeds:
        env_b = make_env(env_cfg, pm, cfg_xai.scenario_id, seed=seed)
        env_a = make_env(env_cfg, pm, cfg_xai.scenario_id, seed=seed)
        env_c = make_env(env_cfg, pm, cfg_xai.scenario_id, seed=seed)
        try:
            groups["baseline"].append(_run_dt_rollout_with_intervention(env_b, dt_model, None, max_steps=max_steps, target_rtg=target_rtg))
            groups["anchor_cf"].append(_run_dt_rollout_with_intervention(env_a, dt_model, anchor_user, max_steps=max_steps, target_rtg=target_rtg))
            groups["control_cf"].append(_run_dt_rollout_with_intervention(env_c, dt_model, control_user, max_steps=max_steps, target_rtg=target_rtg))
        finally:
            env_b.close()
            env_a.close()
            env_c.close()

    def _avg(records, key):
        return float(np.mean([r[key] for r in records])) if records else 0.0

    summary = {}
    for k, recs in groups.items():
        summary[k] = {
            "n": int(len(recs)),
            "return_mean": _avg(recs, "return"),
            "hp_violation_count_mean": _avg(recs, "hp_violation_count"),
            "hp_violation_severity_mean": _avg(recs, "hp_violation_severity"),
            "action_change_proxy_rate_mean": _avg(recs, "action_change_proxy_rate"),
            "kl_mean": _avg(recs, "kl_mean"),
        }

    summary["delta_anchor_minus_baseline"] = {
        "return": summary["anchor_cf"]["return_mean"] - summary["baseline"]["return_mean"],
        "hp_violation_count": summary["anchor_cf"]["hp_violation_count_mean"] - summary["baseline"]["hp_violation_count_mean"],
        "hp_violation_severity": summary["anchor_cf"]["hp_violation_severity_mean"] - summary["baseline"]["hp_violation_severity_mean"],
    }
    summary["delta_control_minus_baseline"] = {
        "return": summary["control_cf"]["return_mean"] - summary["baseline"]["return_mean"],
        "hp_violation_count": summary["control_cf"]["hp_violation_count_mean"] - summary["baseline"]["hp_violation_count_mean"],
        "hp_violation_severity": summary["control_cf"]["hp_violation_severity_mean"] - summary["baseline"]["hp_violation_severity_mean"],
    }
    return {"per_seed": groups, "summary": summary}


def diagnose_anchor_hypothesis(
        obs,
        ppo_model,
        dt_model,
        device,
        slice_assoc,
        crisis_slice,
        crisis_user,
        save_dir,
        control_k=3,
        violation_frames=None,
        frame_limit=10,
):
    ppo_attn_raw, dt_attn_raw = extract_attention_maps(ppo_model, dt_model, obs, device)
    ppo_attn = _to_2d_attn(ppo_attn_raw)
    dt_attn = _to_2d_attn(dt_attn_raw)
    diff = dt_attn - ppo_attn

    gain_per_user = np.mean(diff, axis=0)
    anchor_user = int(np.argmax(gain_per_user))
    anchor_slice = int(np.where(slice_assoc[:, anchor_user] == 1)[0][0])

    # 对照用户：从 anchor 所在 slice 中，选择除 anchor 外 gain 最大的若干用户
    same_slice_users = [int(u) for u in np.where(slice_assoc[anchor_slice] == 1)[0]]
    candidate_controls = [u for u in same_slice_users if u != anchor_user]
    candidate_controls = sorted(candidate_controls, key=lambda u: float(gain_per_user[u]), reverse=True)
    control_users = candidate_controls[:control_k]

    base_emb = _dt_embedding_from_obs(dt_model, obs, device)
    base_dt_logits = _dt_logits_from_obs(dt_model, obs, device)
    base_dt_action = _decode_multidiscrete_action(base_dt_logits, dt_model.action_dims)
    base_ppo_action, _ = ppo_model.predict(obs, deterministic=True)

    interventions = []
    for role, user_idx in [("anchor", anchor_user)] + [("control", u) for u in control_users]:
        obs_cf = _replace_user_with_slice_mean(obs, user_idx, slice_assoc)
        _, dt_cf_raw = extract_attention_maps(ppo_model, dt_model, obs_cf, device)
        dt_cf = _to_2d_attn(dt_cf_raw)
        emb_cf = _dt_embedding_from_obs(dt_model, obs_cf, device)
        dt_logits_cf = _dt_logits_from_obs(dt_model, obs_cf, device)
        dt_action_cf = _decode_multidiscrete_action(dt_logits_cf, dt_model.action_dims)
        ppo_cf_action, _ = ppo_model.predict(obs_cf, deterministic=True)

        interventions.append({
            "role": role,
            "user": int(user_idx),
            "dt_global_l1_change": float(np.mean(np.abs(dt_cf - dt_attn))),
            "dt_crisis_cell_change": float(dt_cf[crisis_slice, crisis_user] - dt_attn[crisis_slice, crisis_user]),
            "dt_anchor_cell_change": float(dt_cf[crisis_slice, anchor_user] - dt_attn[crisis_slice, anchor_user]),
            "dt_embedding_cosine_to_base": _cosine_similarity(base_emb, emb_cf),
            "dt_logits_l1_change": float(np.mean(np.abs(dt_logits_cf - base_dt_logits))),
            "dt_action_changed": bool(np.any(np.array(base_dt_action) != np.array(dt_action_cf))),
            "ppo_action_changed": bool(np.any(np.array(base_ppo_action) != np.array(ppo_cf_action))),
        })

    stability = None
    if violation_frames:
        frames = violation_frames[:max(1, int(frame_limit))]
        per_frame = []
        anchor_counts = {}
        for fr in frames:
            p_attn_raw, d_attn_raw = extract_attention_maps(ppo_model, dt_model, fr["obs"], device)
            p_attn = _to_2d_attn(p_attn_raw)
            d_attn = _to_2d_attn(d_attn_raw)
            frame_gain = np.mean(d_attn - p_attn, axis=0)
            frame_anchor = int(np.argmax(frame_gain))
            anchor_counts[frame_anchor] = anchor_counts.get(frame_anchor, 0) + 1
            per_frame.append({
                "step": int(fr["step"]),
                "slice": int(fr["slice_idx"]),
                "crisis_user": int(fr["user_idx"]),
                "anchor_user": frame_anchor,
                "anchor_gain": float(frame_gain[frame_anchor]),
                "anchor_matches_crisis_user": bool(frame_anchor == int(fr["user_idx"])),
            })

        dominant_user, dominant_count = sorted(anchor_counts.items(), key=lambda x: x[1], reverse=True)[0]
        stability = {
            "frames_analyzed": int(len(frames)),
            "dominant_anchor_user": int(dominant_user),
            "dominant_anchor_frequency": float(dominant_count / len(frames)),
            "anchor_user_counts": {str(k): int(v) for k, v in sorted(anchor_counts.items(), key=lambda x: x[1], reverse=True)},
            "frame_records": per_frame,
        }

        compare_users = [anchor_user] + control_users
        frame_counterfactual = _framewise_counterfactual_panel(
            dt_model=dt_model,
            device=device,
            slice_assoc=slice_assoc,
            frames=frames,
            users=compare_users,
        )
    else:
        frame_counterfactual = None

    diagnosis = {
        "hypothesis": "anchor_baseline",
        "crisis_slice": int(crisis_slice),
        "crisis_user": int(crisis_user),
        "anchor_user": int(anchor_user),
        "anchor_user_slice": int(anchor_slice),
        "anchor_gain_mean_dt_minus_ppo": float(gain_per_user[anchor_user]),
        "top5_users_by_gain": [
            {
                "user": int(u),
                "gain": float(gain_per_user[u]),
                "slice": int(np.where(slice_assoc[:, u] == 1)[0][0]),
            }
            for u in np.argsort(gain_per_user)[-5:][::-1]
        ],
        "baseline": {
            "dt_crisis_attention": float(dt_attn[crisis_slice, crisis_user]),
            "ppo_crisis_attention": float(ppo_attn[crisis_slice, crisis_user]),
            "dt_anchor_attention_on_crisis_slice": float(dt_attn[crisis_slice, anchor_user]),
            "ppo_anchor_attention_on_crisis_slice": float(ppo_attn[crisis_slice, anchor_user]),
        },
        "interventions": interventions,
        "stability": stability,
        "frame_counterfactual": frame_counterfactual,
    }

    with open(os.path.join(save_dir, "anchor_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, ensure_ascii=False, indent=2)
    print(f"🧪 Anchor diagnosis saved: {os.path.join(save_dir, 'anchor_diagnosis.json')}")

    return diagnosis


def plot_heatmaps(ppo_map, dt_map, crisis_slice, crisis_user, save_path):
    """
    绘制对比热力图 (优化版)
    """
    if ppo_map.ndim == 3: ppo_map = np.mean(ppo_map, axis=0)
    if dt_map.ndim == 3: dt_map = np.mean(dt_map, axis=0)

    # 0.04 是均匀分布基准线，0.25 让关注点变成深红
    V_MIN = 0.0
    V_MAX = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    slice_labels = [f"Slice {i}" for i in range(ppo_map.shape[0])]
    ppo_row = ppo_map[crisis_slice]
    dt_row = dt_map[crisis_slice]
    ppo_crisis_val = float(ppo_row[crisis_user])
    dt_crisis_val = float(dt_row[crisis_user])

    # --- PPO ---
    sns.heatmap(ppo_map, ax=axes[0], cmap="Reds", vmin=V_MIN, vmax=V_MAX, cbar=False, linewidths=0.5, linecolor='gray')
    axes[0].set_title(
        f"(a) PPO Attention | crisis u={crisis_user}: {ppo_crisis_val:.3f}",
        fontsize=12, fontweight='bold'
    )
    axes[0].set_xlabel("User Index")
    axes[0].set_ylabel("Query Slice")
    axes[0].set_yticks(np.arange(len(slice_labels)) + 0.5)
    axes[0].set_yticklabels(slice_labels, rotation=0)

    crisis_rect_ppo = plt.Rectangle((crisis_user, crisis_slice), 1, 1, fill=False, edgecolor='blue', lw=3, linestyle='--')
    axes[0].add_patch(crisis_rect_ppo)

    # --- DT ---
    sns.heatmap(dt_map, ax=axes[1], cmap="Reds", vmin=V_MIN, vmax=V_MAX, linewidths=0.5, linecolor='gray', cbar=True)
    axes[1].set_title(
        f"(b) IDT Attention | crisis u={crisis_user}: {dt_crisis_val:.3f}",
        fontsize=12, fontweight='bold'
    )
    axes[1].set_xlabel("User Index")
    axes[1].set_yticks([])

    crisis_rect_dt = plt.Rectangle((crisis_user, crisis_slice), 1, 1, fill=False, edgecolor='blue', lw=3, linestyle='--')
    axes[1].add_patch(crisis_rect_dt)

    for x in range(0, 25, 5):
        axes[0].axvline(x, color='black', lw=1.5)
        axes[1].axvline(x, color='black', lw=1.5)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='blue', lw=3, linestyle='--', label='Crisis user (env-defined)'),
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, "task2_attention_crisis.png"), dpi=300)
    plt.savefig(os.path.join(save_path, "task2_attention_crisis.pdf"), dpi=300)
    print(f"✅ Plot saved to {save_path}")
    print(f"   -> Blue box indicates the Crisis User ({crisis_user}) seen by Crisis Slice ({crisis_slice})")


def execute(cfg_xai, pm, env_cfg):
    env = make_env(env_cfg, pm, cfg_xai.scenario_id)
    ppo, dt = load_models(cfg_xai, env_cfg=env_cfg)
    asset_dir = resolve_project_path(pm, cfg_xai.asset_dir)
    save_dir = resolve_project_path(pm, cfg_xai.output_dir)
    os.makedirs(asset_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if ppo and dt:
        crisis_obs, s_idx, u_idx, selection_meta, violation_frames = hunt_for_crisis(env, dt, cfg_xai)

        if crisis_obs is not None:
            ppo_attn, dt_attn = extract_attention_maps(ppo, dt, crisis_obs, cfg_xai.device)
            np.save(os.path.join(asset_dir, "ppo_attention.npy"), ppo_attn)
            np.save(os.path.join(asset_dir, "dt_attention.npy"), dt_attn)
            slice_assoc = env.components.slices.ue_assoc
            diagnosis = diagnose_anchor_hypothesis(
                obs=crisis_obs,
                ppo_model=ppo,
                dt_model=dt,
                device=cfg_xai.device,
                slice_assoc=slice_assoc,
                crisis_slice=s_idx,
                crisis_user=u_idx,
                save_dir=asset_dir,
                control_k=int(getattr(cfg_xai, "anchor_control_users", 3)),
                violation_frames=violation_frames,
                frame_limit=int(getattr(cfg_xai, "anchor_diagnosis_frames", 10)),
            )
            control_user = next((r["user"] for r in diagnosis["interventions"] if r["role"] == "control"), diagnosis["anchor_user"])
            rollout_impact = run_rollout_impact(
                env_cfg=env_cfg,
                pm=pm,
                cfg_xai=cfg_xai,
                dt_model=dt,
                anchor_user=diagnosis["anchor_user"],
                control_user=control_user,
            )
            diagnosis["rollout_impact"] = rollout_impact
            with open(os.path.join(asset_dir, "anchor_diagnosis.json"), "w", encoding="utf-8") as f:
                json.dump(diagnosis, f, ensure_ascii=False, indent=2)
            print(f"🧪 Rollout impact saved: {os.path.join(asset_dir, 'anchor_diagnosis.json')}")
            with open(os.path.join(asset_dir, "crisis_meta.json"), "w", encoding="utf-8") as f:
                ppo_2d = np.mean(ppo_attn, axis=0) if ppo_attn.ndim == 3 else ppo_attn
                dt_2d = np.mean(dt_attn, axis=0) if dt_attn.ndim == 3 else dt_attn
                ppo_row = ppo_2d[s_idx]
                dt_row = dt_2d[s_idx]
                frame_cf = diagnosis.get("frame_counterfactual") or []
                anchor_cf = next((r for r in frame_cf if r.get("user") == diagnosis["anchor_user"]), None)
                json.dump({
                    "scenario_id": int(cfg_xai.scenario_id),
                    "crisis_threshold": float(cfg_xai.crisis_threshold),
                    "crisis_slice": int(s_idx),
                    "crisis_user": int(u_idx),
                    "search_max_steps": int(getattr(cfg_xai, "search_max_steps", 200)),
                    "fallback_to_peak": bool(getattr(cfg_xai, "fallback_to_peak", True)),
                    **selection_meta,
                    "ppo_crisis_attention": float(ppo_row[u_idx]),
                    "dt_crisis_attention": float(dt_row[u_idx]),
                    "ppo_top_user_on_crisis_slice": int(np.argmax(ppo_row)),
                    "dt_top_user_on_crisis_slice": int(np.argmax(dt_row)),
                    "ppo_top_attention_on_crisis_slice": float(np.max(ppo_row)),
                    "dt_top_attention_on_crisis_slice": float(np.max(dt_row)),
                    "anchor_user": int(diagnosis["anchor_user"]),
                    "anchor_gain_mean_dt_minus_ppo": float(diagnosis["anchor_gain_mean_dt_minus_ppo"]),
                    "violation_frames_collected": int(len(violation_frames)),
                    "dominant_anchor_user_across_frames": int(diagnosis["stability"]["dominant_anchor_user"]) if diagnosis["stability"] else -1,
                    "dominant_anchor_frequency": float(diagnosis["stability"]["dominant_anchor_frequency"]) if diagnosis["stability"] else 0.0,
                    "anchor_cf_action_change_rate": float(anchor_cf["action_change_rate"]) if anchor_cf else 0.0,
                    "anchor_cf_kl_mean": float(anchor_cf["kl_mean"]) if anchor_cf else 0.0,
                    "rollout_delta_anchor_return": float(rollout_impact["summary"]["delta_anchor_minus_baseline"]["return"]),
                    "rollout_delta_anchor_violation_count": float(rollout_impact["summary"]["delta_anchor_minus_baseline"]["hp_violation_count"]),
                    "rollout_delta_anchor_violation_severity": float(rollout_impact["summary"]["delta_anchor_minus_baseline"]["hp_violation_severity"]),
                }, f, ensure_ascii=False, indent=2)
            plot_heatmaps(ppo_attn, dt_attn, s_idx, u_idx, save_path=save_dir)

    env.close()


def attention_entropy(attn_map):
    """
    H2 实验：计算 Attention 分布的 Shannon 熵，衡量注意力的聚焦程度。

    低熵 → 注意力集中（聚焦危机用户）；高熵 → 注意力分散（均匀关注）。

    Args:
        attn_map: ndarray, 形状 [n_heads, n_queries, n_keys] 或 [n_queries, n_keys]。
                  若为多头，先对 heads 取平均。

    Returns:
        entropy_per_query: ndarray [n_queries]，每个 query（slice）的平均熵。
        mean_entropy: float，全局平均熵（用于论文标注）。
    """
    if attn_map.ndim == 3:
        # 多头：对 head 维度取平均
        a = np.mean(attn_map, axis=0)   # [n_queries, n_keys]
    else:
        a = attn_map                     # [n_queries, n_keys]

    # 数值稳定：确保概率归一化
    a = np.clip(a, 1e-12, None)
    a = a / a.sum(axis=-1, keepdims=True)

    # Shannon 熵：H = -sum(p * log(p))
    entropy_per_query = -np.sum(a * np.log(a), axis=-1)   # [n_queries]
    mean_entropy = float(np.mean(entropy_per_query))

    return entropy_per_query, mean_entropy


def compare_attention_entropy(ppo_attn, dt_attn, save_path=None):
    """
    H2 辅助函数：比较 PPO 和 IDT 的注意力熵，并可视化。

    Args:
        ppo_attn: PPO attention map, ndarray.
        dt_attn: IDT attention map, ndarray.
        save_path: 若提供，保存 entropy 对比条形图。

    Returns:
        dict with ppo_entropy / dt_entropy (per-query and mean).
    """
    ppo_per_q, ppo_mean = attention_entropy(ppo_attn)
    dt_per_q, dt_mean = attention_entropy(dt_attn)

    print(f"Attention Entropy — PPO: {ppo_mean:.4f} | IDT: {dt_mean:.4f}")

    if save_path is not None:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(ppo_per_q))
        ax.bar(x - 0.2, ppo_per_q, 0.4, label=f"PPO (mean={ppo_mean:.3f})", color='#d62728', alpha=0.85)
        ax.bar(x + 0.2, dt_per_q, 0.4, label=f"Proposed IDT (mean={dt_mean:.3f})", color='#1f77b4', alpha=0.85)
        ax.set_xlabel("Slice Index")
        ax.set_ylabel("Attention Entropy (bits)")
        ax.set_title("H2: Attention Entropy Comparison", fontweight='bold')
        ax.legend()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for ext in ('png', 'pdf'):
            plt.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Entropy comparison saved to {save_path}")

    return {
        "ppo_entropy_per_query": ppo_per_q.tolist(),
        "ppo_mean_entropy": ppo_mean,
        "dt_entropy_per_query": dt_per_q.tolist(),
        "dt_mean_entropy": dt_mean,
    }


if __name__ == "__main__":
    pass
