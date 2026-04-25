import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# === Local Imports ===
# 确保这些文件在同一目录下或在 PYTHONPATH 中
from src.basic_apis.dt_utils.dataset import HierarchicalDTDataset
from src.basic_apis.dt_utils.model_ha_dt import HierarchicalStateEncoder, build_decision_model
from src.basic_apis.dt_utils.action_dims import resolve_action_dims_from_env_cfg
from src.basic_apis.network_slicing_business.path_context import PathContext
from src.basic_apis.ppo.ppo_ha_weighted.hierarchical_slicing_env import HierarchicalSlicingEnv
from src.basic_apis.general_utils import pad_stack_tensor
from src.basic_apis.asset_utils import build_versioned_run_dir, update_latest_symlink, ensure_clean_dir, ensure_dir


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


def _compute_step_metrics(info, num_slices=None):
    """与 test.py 对齐的单步指标计算。
    返回: (hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active)
    """
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    resolved = _infer_num_slices_from_info(info) if num_slices is None else num_slices
    for s_idx in range(resolved):
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
        mean_drift = float(np.mean(drifts))
        if mean_drift < 0:
            if is_hp:
                hp_dist += mean_drift
                hp_viols += 1
            else:
                nhp_dist += mean_drift
                nhp_viols += 1
    return hp_dist, nhp_dist, hp_viols, nhp_viols, hp_active, nhp_active


def _normalize_obs(raw, feat_key, obs_stats):
    """Normalize a raw observation array, matching HierarchicalDTDataset logic."""
    if obs_stats is None:
        return raw
    stat_key = feat_key.split('_')[0]
    if stat_key not in obs_stats:
        return raw
    mean = obs_stats[stat_key]['mean']
    std = obs_stats[stat_key]['std']
    safe_std = std.copy()
    safe_std[safe_std < 1e-2] = 1.0
    normed = (raw - mean) / safe_std
    return np.clip(normed, -5.0, 5.0).astype(np.float32)


def make_env(cfg, path_context, seed=0):
    """
    环境工厂函数，用于闭环评估
    """

    def _init():
        # 1. 复制配置以免污染全局配置
        env_config = cfg.env_settings.copy()

        # 2. 强制设为评估模式 (通常使用验证集对应的场景)
        env_config.mode = 'evaluating'

        # 3. 兜底逻辑：确保 scenario list 存在
        scenario_mode = env_config.scenario_mode
        target_cfg = env_config[scenario_mode]['evaluating']
        if target_cfg.active_scenario_list is None:
            # 如果配置为空，默认使用 yaml 定义的范围 (例如 80-84)
            target_cfg.active_scenario_list = [0]

        # 初始化环境
        env = HierarchicalSlicingEnv(env_config, np.random.default_rng(seed), path_context)
        return env

    return _init



def evaluate_in_env(model, env_fn, context_len, target_rtg, rtg_scale, device, action_dims,
                     obs_stats=None):
    """
    在真实环境中进行闭环评估 (Closed-loop Inference)

    Args:
        target_rtg: 必须是 Raw Value (例如 -60000.0)，不要预先做 SymLog！
        action_dims: 动作维度列表，例如 [11, 3, 3, 3, 3, 3]
        obs_stats: 观测归一化统计量（与 HierarchicalDTDataset 一致），None 则不归一化

    Returns:
        dict: {reward, hp_dist, nhp_dist, hp_viols, nhp_viols}
    """
    env = env_fn()
    obs, _ = env.reset()

    model.eval()
    total_reward = 0
    done = False
    ep_hp_dist = ep_nhp_dist = 0.0
    ep_hp_viols = ep_nhp_viols = 0
    ep_hp_active = ep_nhp_active = 0

    # === [修正] Context 初始化 ===
    # 动作是 Multi-Discrete，长度为 len(action_dims) (即 6)
    # 类型必须是 Long
    dummy_action = torch.zeros(len(action_dims), dtype=torch.long, device=device)

    history = {
        "inter": [], "intra": [], "global": [],
        "actions": [dummy_action],
        "rtg": [], "timesteps": []
    }

    # curr_rtg_val 维护的是物理世界的 Raw Reward
    curr_rtg_val = target_rtg
    curr_step = 0

    with torch.no_grad():
        while not done:
            # === 1. Observation 处理（与训练时 Dataset 归一化保持一致）===
            t_inter = torch.from_numpy(_normalize_obs(obs['inter_feat'], 'inter_feat', obs_stats)).float().to(device)
            t_intra = torch.from_numpy(_normalize_obs(obs['intra_feat'], 'intra_feat', obs_stats)).float().to(device)
            t_glob = torch.from_numpy(_normalize_obs(obs['global_feat'], 'global_feat', obs_stats)).float().to(device)

            history["inter"].append(t_inter)
            history["intra"].append(t_intra)
            history["global"].append(t_glob)

            # RTG SymLog 变换（与 Dataset 训练时一致：只做 sym-log，不除以 rtg_scale）
            rtg_symlog = np.sign(curr_rtg_val) * np.log1p(np.abs(curr_rtg_val))
            t_rtg = torch.tensor([rtg_symlog], dtype=torch.float32, device=device)
            history["rtg"].append(t_rtg)

            # Timestep
            t_step = torch.tensor([curr_step], dtype=torch.long, device=device)
            history["timesteps"].append(t_step)

            # === 2. 上下文截断 ===
            if len(history["inter"]) > context_len:
                for k in ["inter", "intra", "global", "rtg", "timesteps", "actions"]:
                    history[k].pop(0)

            # === 3. Stack & Pad (保持不变) ===
            in_inter = pad_stack_tensor(history["inter"], context_len, None, device)
            in_intra = pad_stack_tensor(history["intra"], context_len, None, device)
            in_global = pad_stack_tensor(history["global"], context_len, 2, device)
            in_rtg = pad_stack_tensor(history["rtg"], context_len, 1, device)
            in_steps = pad_stack_tensor(history["timesteps"], context_len, 1, device).squeeze(-1).long()

            # Action 已经是 Long Tensor [K, 6]，直接 stack
            # pad_stack_tensor 需要适配 long 类型，或者手动处理 action stack
            # 这里复用 pad_stack_tensor，假设它能处理 int64，或者这里手动 pad
            # 为了稳健，手动处理 Actions Padding:
            act_seq = torch.stack(history["actions"])  # [T, 6]
            if act_seq.shape[0] < context_len:
                pad_len = context_len - act_seq.shape[0]
                pad = torch.zeros((pad_len, len(action_dims)), dtype=torch.long, device=device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_actions = act_seq.unsqueeze(0)  # [1, K, 6]

            states_in = {'inter': in_inter, 'intra': in_intra, 'global': in_global}

            # Mask
            real_len = min(len(history["rtg"]), context_len)
            mask = torch.zeros((1, context_len), device=device)
            mask[0, -real_len:] = 1.0

            # === 4. 模型推理 ===
            action_preds = model(
                states=states_in,
                actions=in_actions,
                returns=in_rtg,
                timesteps=in_steps,
                attention_mask=mask
            )

            # === [修正] Multi-Discrete 动作解析 ===
            # 取最后一个时间步 Logits: [Total_Logits]
            last_logits = action_preds[0, -1, :]

            pred_actions = []
            start_idx = 0
            for dim_size in action_dims:
                end_idx = start_idx + dim_size
                dim_logit = last_logits[start_idx:end_idx]

                # Argmax (确定性策略)
                dim_act = torch.argmax(dim_logit).item()
                pred_actions.append(dim_act)
                start_idx = end_idx

            # 转为 numpy (Int32) 传给环境
            action = np.array(pred_actions, dtype=np.int32)

            # === 5. 环境交互 ===
            # 扩展维度适配 VecEnv [1, 6]
            # vec_action = np.expand_dims(action, axis=0)
            # next_obs, reward, done, _, _ = env.step(vec_action)

            next_obs, reward, done, _, info = env.step(action)

            # === 6. 更新状态 ===
            action_tensor = torch.tensor(action, dtype=torch.long, device=device)
            history["actions"].append(action_tensor)

            curr_rtg_val -= reward
            total_reward += reward

            # 累计真实 metric（与 test.py 对齐）
            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(info)
            ep_hp_dist += hp_d
            ep_nhp_dist += nhp_d
            ep_hp_viols += hp_v
            ep_nhp_viols += nhp_v
            ep_hp_active += hp_a
            ep_nhp_active += nhp_a

            obs = next_obs
            curr_step += 1

    env.close()
    return {
        "reward": total_reward,
        "hp_dist": ep_hp_dist / ep_hp_active if ep_hp_active > 0 else 0.0,
        "nhp_dist": ep_nhp_dist / ep_nhp_active if ep_nhp_active > 0 else 0.0,
        "hp_viols": ep_hp_viols / ep_hp_active if ep_hp_active > 0 else 0.0,
        "nhp_viols": ep_nhp_viols / ep_nhp_active if ep_nhp_active > 0 else 0.0,
    }


def train(cfg: DictConfig, path_context: PathContext):
    """
    DT 训练主入口
    """
    dt_cfg = cfg.train_dt
    device = dt_cfg.device
    print(f"🚀 DT Training Device: {device}")

    # 1. 路径准备
    base_dir = dt_cfg.dataset.base_dir
    train_path = os.path.join(base_dir, dt_cfg.dataset.train_path)
    val_path = os.path.join(base_dir, dt_cfg.dataset.val_path)
    meta_path = os.path.join(base_dir, dt_cfg.dataset.meta_path)
    asset_cfg = dt_cfg.saving.get("asset", None)
    if asset_cfg and bool(asset_cfg.get("use_versioned_runs", False)):
        model_root = str(asset_cfg.model_root)
        run_id_cfg = asset_cfg.get("run_id", "auto")
        run_id = None if str(run_id_cfg) == "auto" else str(run_id_cfg)
        run_dir = build_versioned_run_dir(model_root, run_id=run_id)
        if bool(asset_cfg.get("clean_before_run", False)):
            ensure_clean_dir(run_dir)
        else:
            ensure_dir(run_dir)
        save_dir = Path(run_dir)
        print(f"[Asset] versioned run dir: {save_dir}")
    else:
        save_dir = Path(dt_cfg.saving.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 数据集加载
    print(f"📥 Loading Train Dataset: {train_path}")
    train_dataset = HierarchicalDTDataset(
        train_path,
        context_len=dt_cfg.model.context_len,
        rtg_scale=dt_cfg.dataset.rtg_scale,
        metadata_path=meta_path
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=dt_cfg.optimizer.batch_size,
        shuffle=True,
        num_workers=dt_cfg.dataset.num_workers,
        pin_memory=True
    )

    val_loader = None
    if os.path.exists(val_path):
        print(f"📥 Loading Val Dataset: {val_path}")
        val_dataset = HierarchicalDTDataset(
            val_path,
            context_len=dt_cfg.model.context_len,
            rtg_scale=dt_cfg.dataset.rtg_scale,
            metadata_path=meta_path
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=dt_cfg.optimizer.batch_size,
            shuffle=False,
            num_workers=dt_cfg.dataset.num_workers
        )

    # 提取观测归一化统计量，供闭环评估时使用（与 Dataset 训练时一致）
    obs_stats = train_dataset.stats if train_dataset.normalize else None

    # 3. 模型初始化
    state_encoder = HierarchicalStateEncoder(embed_dim=dt_cfg.model.embed_dim)

    # 从环境配置动态推导 action dims，避免 M=11 硬编码。
    action_dims = resolve_action_dims_from_env_cfg(cfg.environment, intra_mode_count=3)

    model = build_decision_model(
        model_cfg=dt_cfg.model,
        state_encoder=state_encoder,
        action_dims=action_dims,
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f"🔥 Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # 优化器配置
    optimizer = AdamW(
        model.parameters(),
        lr=dt_cfg.optimizer.learning_rate,
        weight_decay=dt_cfg.optimizer.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=dt_cfg.optimizer.epochs
    )

    # loss_fn = torch.nn.MSELoss()

    # === [修改] Loss Function ===
    # ignore_index=-100 自动忽略 padding 部分
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    use_amp = dt_cfg.get("use_amp", False) and "cuda" in str(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"⚡ AMP: {'ON' if use_amp else 'OFF'}  |  batch_size: {dt_cfg.optimizer.batch_size}")

    best_val_loss = float('inf')
    best_eval_key = None  # 字典序 (hp_viols, -hp_dist, nhp_viols, -nhp_dist)，越小越好

    training_history = {
        "train_loss": [],
        "val_loss": [],
        "eval_reward": [],
        "eval_metrics": [],
        "eval_epoch": [],
    }

    # 4. 训练循环
    print(f"🏁 Starting Training for {dt_cfg.optimizer.epochs} epochs...")

    for epoch in range(dt_cfg.optimizer.epochs):
        model.train()
        train_loss = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{dt_cfg.optimizer.epochs}")
        for batch in pbar:
            # 数据搬运
            states = {
                'inter': batch['inter'].to(device),
                'intra': batch['intra'].to(device),
                'global': batch['global'].to(device)
            }
            actions_in = batch['actions'].to(device)
            actions_target = batch['action_targets'].to(device)

            rtg = batch['rtg'].to(device)
            timesteps = batch['timesteps'].to(device)
            mask = batch['mask'].to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                action_preds = model(
                    states=states,
                    actions=actions_in,
                    returns=rtg,
                    timesteps=timesteps,
                    attention_mask=mask
                )

                loss = 0.0
                start_idx = 0
                preds_flat = action_preds.reshape(-1, action_preds.shape[-1])
                targets_flat = actions_target.reshape(-1, actions_target.shape[-1])

                for i, dim_size in enumerate(action_dims):
                    end_idx = start_idx + dim_size
                    dim_logits = preds_flat[:, start_idx:end_idx]
                    dim_target = targets_flat[:, i]
                    loss += loss_fn(dim_logits, dim_target)
                    start_idx = end_idx

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            scaler.step(optimizer)
            scaler.update()

            train_loss.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.2e}"})

        scheduler.step()
        avg_train_loss = np.mean(train_loss)

        # 5. Open-Loop Validation
        avg_val_loss = 0.0
        if val_loader:
            model.eval()
            val_losses = []

            # 定义钩子函数：用于捕获每一层的输出
            debug_activations = {}

            def get_activation(name):
                def hook(model, input, output):
                    # 如果输出包含 NaN，记录下来
                    if isinstance(output, torch.Tensor):
                        if torch.isnan(output).any():
                            debug_activations[name] = "NaN Detected!"
                        elif torch.isinf(output).any():
                            debug_activations[name] = "Inf Detected!"
                    elif isinstance(output, tuple):
                        # Transformer 有时返回 tuple
                        if torch.isnan(output[0]).any():
                            debug_activations[name] = "NaN Detected (Tuple)!"

                return hook

            print("🔍 [Debug] Validation Started with NaN Hooks...")

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    debug_activations = {}

                    states = {
                        'inter': batch['inter'].to(device),
                        'intra': batch['intra'].to(device),
                        'global': batch['global'].to(device)
                    }
                    actions_in = batch['actions'].to(device)
                    actions_target = batch['action_targets'].to(device)
                    rtg = batch['rtg'].to(device)
                    timesteps = batch['timesteps'].to(device)
                    mask = batch['mask'].to(device)
                    if timesteps.max() >= 1000:
                        print(f"⚠️ [Batch {batch_idx}] Timestep Out of Bounds! Max: {timesteps.max()}")

                    if torch.isnan(actions_target).any() or torch.isinf(actions_target).any():
                        print(f"💀 [FATAL] Batch {batch_idx}: TARGETS contain NaN/Inf!")
                        break

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        preds = model(states, actions_in, rtg, timesteps, mask)

                    current_loss = 0.0
                    start_idx = 0
                    preds_flat = preds.float().reshape(-1, preds.shape[-1])
                    targets_flat = actions_target.reshape(-1, actions_target.shape[-1])

                    for i, dim_size in enumerate(action_dims):
                        end_idx = start_idx + dim_size
                        dim_logits = preds_flat[:, start_idx:end_idx]
                        dim_target = targets_flat[:, i]
                        current_loss += loss_fn(dim_logits, dim_target).item()
                        start_idx = end_idx

                    val_losses.append(current_loss)

            # 计算平均
            if len(val_losses) > 0:
                avg_val_loss = np.mean(val_losses)
            else:
                avg_val_loss = 0.0  # 标记失败


            # Val loss 仅用于监控，不用于模型选择（best model 由闭环 metric 决定）
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

        training_history["train_loss"].append(float(avg_train_loss))
        training_history["val_loss"].append(float(avg_val_loss))

        log_msg = f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.2e}"
        if val_loader:
            log_msg += f" | Val Loss: {avg_val_loss:.2e}"
        print(log_msg)
        _raw = model.module if hasattr(model, 'module') else model
        torch.save(_raw.state_dict(), save_dir / f"epoch_{epoch}.pth")

        if dt_cfg.evaluation.active and (epoch + 1) % dt_cfg.evaluation.eval_freq_epoch == 0:
            target_raw = dt_cfg.evaluation.target_rtg

            print(f"🔄 Running Eval in Environment (Target RTG Raw: {target_raw})...")
            ep_results = []

            for i in range(dt_cfg.evaluation.n_eval_episodes):
                env_fn = make_env(cfg.environment, path_context, seed=10000 + epoch + i)
                result = evaluate_in_env(
                    model, env_fn,
                    context_len=dt_cfg.model.context_len,
                    target_rtg=target_raw,
                    rtg_scale=dt_cfg.dataset.rtg_scale,
                    device=device,
                    action_dims=action_dims,
                    obs_stats=obs_stats,
                )
                ep_results.append(result)

            avg_metrics = {
                k: float(np.mean([r[k] for r in ep_results]))
                for k in ("reward", "hp_dist", "nhp_dist", "hp_viols", "nhp_viols")
            }
            training_history["eval_reward"].append(avg_metrics["reward"])
            training_history["eval_metrics"].append(avg_metrics)
            training_history["eval_epoch"].append(epoch + 1)

            # 字典序模型选择：(hp_viols, -hp_dist, nhp_viols, -nhp_dist)，越小越好
            eval_key = (
                avg_metrics["hp_viols"],
                -avg_metrics["hp_dist"],
                avg_metrics["nhp_viols"],
                -avg_metrics["nhp_dist"],
            )
            if best_eval_key is None or eval_key < best_eval_key:
                best_eval_key = eval_key
                _raw = model.module if hasattr(model, 'module') else model
                torch.save(_raw.state_dict(), save_dir / "best_dt_model.pth")
                print(f"   ✅ New best model saved! key={tuple(f'{v:.4f}' for v in eval_key)}")

            print(f"   -> Reward: {avg_metrics['reward']:.1f} | "
                  f"HP(v={avg_metrics['hp_viols']:.3f}, d={avg_metrics['hp_dist']:.4f}) | "
                  f"NHP(v={avg_metrics['nhp_viols']:.3f}, d={avg_metrics['nhp_dist']:.4f})")

    # 保存最终模型
    _raw = model.module if hasattr(model, 'module') else model
    torch.save(_raw.state_dict(), save_dir / "final_dt_model.pth")

    import json
    history_path = save_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"📊 Training history saved to {history_path}")

    print(f"✅ Training Finished. Final model saved to {save_dir}")

    if asset_cfg and bool(asset_cfg.get("use_versioned_runs", False)) and bool(asset_cfg.get("update_latest", True)):
        model_root = str(asset_cfg.model_root)
        link_path = update_latest_symlink(model_root, str(save_dir))
        print(f"[Asset] updated latest symlink: {link_path}")


if __name__ == "__main__":
    config_path = "hierarchical_env.yaml"
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        work_dir = os.getcwd()
        pm = PathContext(work_dir)
        train(cfg, pm)
    else:
        print(f"❌ Config file {config_path} not found.")