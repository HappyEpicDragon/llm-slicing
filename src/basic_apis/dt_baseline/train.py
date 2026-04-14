import os
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

# === Local Imports ===
from src.basic_apis.dt_baseline.model_baseline_dt import DecisionTransformerBaseline, BaselineStateEncoder
from src.basic_apis.dt_baseline.dataset import BaselineDTDataset
from src.basic_apis.network_slicing_business.path_manager import PathManager
# 复用 Ray 环境创建逻辑 (需要确保这些文件在路径中)
from src.basic_apis.ppo.ppo_baseline.env_ray import env_creator


# ==============================================================================
# 1. Baseline 专用数据集 (Flat Obs + Hybrid Actions)
# ==============================================================================

# ==============================================================================
# 2. 闭环评估逻辑 (Closed-Loop Evaluation)
# ==============================================================================

def _compute_step_metrics(info, num_slices=5):
    """与 test.py 对齐的单步指标计算。"""
    flat_info = info if isinstance(info, dict) else {}
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
        is_hp = flat_info.get(f"meta/slice_{s_idx}_priority", 0) > 0
        drifts = [float(flat_info.get(f"drift/slice_{s_idx}_{met}", 0.0))
                  for met in ("thr", "rel", "lat")
                  if flat_info.get(f"meta/slice_{s_idx}_{met}_req", 0) > 0]
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


def make_ray_env(cfg, seed=0):
    """
    创建 Ray 兼容的评估环境 (不启动 Ray 集群，直接作为 Python 对象使用)
    """
    env_config = OmegaConf.to_container(cfg.environment, resolve=True)
    # 强制覆盖为评估模式
    mode = 'evaluating'
    scenario_mode = cfg.env_updates.scenario_mode

    env_config['mode'] = mode
    env_config['scenario_mode'] = scenario_mode
    # 确保配置正确读取
    if 'evaluating' in cfg.env_updates[scenario_mode]:
        target_list = cfg.env_updates[scenario_mode].evaluating.active_scenario_list
        env_config[scenario_mode]['evaluating']['active_scenario_list'] = target_list
        # 这里特别重要：Ray env_creator 读取的是 state_config 里的 active_scenario_list
        # 所以必须手动覆盖进去
        env_config['state_config']['active_scenario_list'] = target_list

    env_config['seed'] = seed

    # 本地创建环境实例
    return env_creator(env_config, only_env=True)


def evaluate_in_env_baseline(model, env, context_len, target_rtg_raw, rtg_scale,
                             device, obs_mean, obs_std):
    """
    Baseline DT 的闭环推理循环
    """
    # Ray Env Reset -> (obs, info)
    obs_dict, _ = env.reset()

    model.eval()
    total_reward = 0
    done = False
    curr_step = 0
    ep_hp_dist = ep_nhp_dist = 0.0
    ep_hp_viols = ep_nhp_viols = 0
    ep_hp_active = ep_nhp_active = 0

    # 动态 RTG 计算
    curr_rtg_val = target_rtg_raw

    # Context Buffers
    # Action Dim = 10 (5 Cont + 5 Disc)
    dummy_action = torch.zeros(10, device=device)

    history = {
        "obs": [],  # List of [Obs_Dim]
        "actions": [dummy_action],
        "rtg": [],
        "timesteps": []
    }

    # 扁平化辅助函数 (需与 Dataset 收集时逻辑一致)
    def flatten_obs(obs_d):
        flat = []
        if "player_0" in obs_d:
            p0 = obs_d["player_0"]
            flat.append(p0["observations"] if isinstance(p0, dict) else p0)
        for i in range(1, 6):
            key = f"player_{i}"
            if key in obs_d:
                p = obs_d[key]
                flat.append(p["observations"] if isinstance(p, dict) else p)
        return np.concatenate(flat, axis=0).astype(np.float32)

    with torch.no_grad():
        while not done:
            # 1. 处理 Observation
            flat_obs = flatten_obs(obs_dict)
            # 标准化 (非常重要!)
            norm_obs = np.clip((flat_obs - obs_mean) / (obs_std + 1e-6), -5.0, 5.0)

            t_obs = torch.from_numpy(norm_obs).float().to(device)
            history["obs"].append(t_obs)

            # 2. 处理 RTG (SymLog)
            # RTG SymLog 变换（与 Dataset 训练时一致：只做 sym-log，不除以 rtg_scale）
            rtg_input = np.sign(curr_rtg_val) * np.log1p(np.abs(curr_rtg_val))
            t_rtg = torch.tensor([rtg_input], dtype=torch.float32, device=device)
            history["rtg"].append(t_rtg)

            # 3. Timestep
            t_step = torch.tensor([curr_step], dtype=torch.long, device=device)
            history["timesteps"].append(t_step)

            # 4. 截断上下文
            if len(history["obs"]) > context_len:
                for k in history: history[k].pop(0)

            # 5. Padding & Stacking
            # Helper: List[Tensor] -> [1, K, Dim]
            def stack_pad(lst, dim=None):
                seq = torch.stack(lst)
                curr_l = seq.shape[0]
                if curr_l < context_len:
                    pad_s = (context_len - curr_l, *seq.shape[1:])
                    pad = torch.zeros(pad_s, device=device)
                    seq = torch.cat([pad, seq], dim=0)
                if seq.ndim == 1: seq = seq.unsqueeze(-1)
                return seq.unsqueeze(0)

            in_obs = stack_pad(history["obs"])
            in_rtg = stack_pad(history["rtg"])
            in_time = stack_pad(history["timesteps"]).squeeze(-1)

            # 处理 Action Stack
            act_seq = torch.stack(history["actions"])
            if act_seq.shape[0] < context_len:
                pad = torch.zeros((context_len - act_seq.shape[0], 10), device=device)
                act_seq = torch.cat([pad, act_seq], dim=0)
            in_act = act_seq.unsqueeze(0)

            # Mask
            real_len = min(len(history["obs"]), context_len)
            mask = torch.zeros((1, context_len), device=device)
            mask[0, -real_len:] = 1.0

            # 6. 推理
            # outputs: (preds_cont, preds_disc_logits)
            p_cont, p_disc_logits = model(in_obs, in_act, in_rtg, in_time, mask)

            # 取最后一步
            last_cont = p_cont[0, -1, :]  # [5]
            last_logits = p_disc_logits[0, -1]  # [5, 3]

            # 解析离散动作 (Argmax)
            last_disc = torch.argmax(last_logits, dim=1).float()  # [5]

            # 7. 构造环境动作字典 (Ray Format)
            # Baseline PPO 期望的动作是:
            # player_0: Box(5) -> 对应 last_cont
            # player_1~5: Discrete(3) -> 对应 last_disc[0]...[4]

            action_dict = {}
            action_dict["player_0"] = last_cont.cpu().numpy()

            disc_np = last_disc.cpu().numpy().astype(int)
            for i in range(5):
                action_dict[f"player_{i + 1}"] = disc_np[i]

            # 8. Step
            obs_dict, rewards, terminated, truncated, infos = env.step(action_dict)

            done = terminated["__all__"] or truncated["__all__"]

            # 9. 更新历史
            hybrid_act = torch.cat([last_cont, last_disc], dim=0)
            history["actions"].append(hybrid_act)

            r = rewards.get("player_0", 0.0)
            curr_rtg_val -= r
            total_reward += r

            # 累计真实 metric
            step_info = infos.get("player_0", infos) if isinstance(infos, dict) else {}
            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(step_info)
            ep_hp_dist += hp_d
            ep_nhp_dist += nhp_d
            ep_hp_viols += hp_v
            ep_nhp_viols += nhp_v
            ep_hp_active += hp_a
            ep_nhp_active += nhp_a

            curr_step += 1

    env.close()
    return {
        "reward": total_reward,
        "hp_dist": ep_hp_dist / ep_hp_active if ep_hp_active > 0 else 0.0,
        "nhp_dist": ep_nhp_dist / ep_nhp_active if ep_nhp_active > 0 else 0.0,
        "hp_viols": ep_hp_viols / ep_hp_active if ep_hp_active > 0 else 0.0,
        "nhp_viols": ep_nhp_viols / ep_nhp_active if ep_nhp_active > 0 else 0.0,
    }


# ==============================================================================
# 3. 主训练循环
# ==============================================================================

def train(cfg: DictConfig, path_manager: PathManager):
    dt_cfg = cfg.train_dt
    device = dt_cfg.device
    print(f"🚀 DT-Baseline Training Device: {device}")

    # 1. 路径与数据加载
    base_dir = dt_cfg.dataset.base_dir
    train_path = os.path.join(base_dir, "training")  # 假设 Baseline 数据放在这
    meta_path = os.path.join(base_dir, "metadata.json")
    save_dir = Path(dt_cfg.saving.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Loading Dataset: {train_path}")
    train_dataset = BaselineDTDataset(
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

    # 2. 模型初始化
    obs_dim = train_dataset.obs_mean.shape[0]
    print(f"🧠 Init Model with State Dim: {obs_dim}")

    # Encoder: Flatten Input -> Hidden
    state_encoder = BaselineStateEncoder(input_dim=obs_dim, embed_dim=dt_cfg.model.embed_dim)

    model = DecisionTransformerBaseline(
        state_encoder=state_encoder,
        state_dim=obs_dim,
        hidden_size=dt_cfg.model.embed_dim,
        max_length=dt_cfg.model.context_len,
        # 这里的参数根据 PPO Baseline 设定 (1个Player0 + 5个Intra Agents)
        act_dim_cont=5,
        act_num_discrete=5,
        act_discrete_card=3
    ).to(device)

    # 3. 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=dt_cfg.optimizer.learning_rate,
        weight_decay=dt_cfg.optimizer.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=dt_cfg.optimizer.epochs
    )

    # 4. 损失函数 (Hybrid)
    loss_fn_cont = nn.MSELoss()
    loss_fn_disc = nn.CrossEntropyLoss(ignore_index=-100)  # -100 for padding if needed

    best_loss = float('inf')

    # 5. 训练循环
    for epoch in range(dt_cfg.optimizer.epochs):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            obs = batch['obs'].to(device)

            # === [修改点] Action Input ===
            # 原代码: 手动 shift
            # act_raw = batch['actions'].to(device)
            # act_in = torch.cat([torch.zeros...], dim=1)

            # 新代码: 直接使用 Dataset 返回的 'actions' (Input) 和 'action_targets' (Label)
            act_in = batch['actions'].to(device)  # 已经是 a_{t-1}
            act_target = batch['action_targets'].to(device)  # 已经是 a_t

            rtg = batch['rtg'].to(device)
            time = batch['timesteps'].to(device)
            mask = batch['mask'].to(device)

            # Forward
            preds_cont, preds_disc_logits = model(obs, act_in, rtg, time, mask)

            # === Hybrid Loss Calculation ===
            # Mask 展平 -> [B*K]
            mask_flat = mask.view(-1).bool()

            # 1. Continuous Loss (Player 0)
            # Target: act_target[:, :, :5] -> [B, K, 5]
            target_cont = act_target[:, :, :5]
            # 只计算 Mask=1 的部分
            loss_c = loss_fn_cont(preds_cont[mask.bool()], target_cont[mask.bool()])

            # 2. Discrete Loss (Player 1-5)
            # Target: act_target[:, :, 5:] -> [B, K, 5] -> Long
            # Preds: [B, K, 5, 3]
            target_disc = act_target[:, :, 5:].long()

            # Flatten for CE: Preds [N, 3], Target [N]
            # 需要把 B, K, 5 展平
            # 只取 Mask=1 的时间步
            # preds_disc_logits: [B, K, 5, 3] -> mask select -> [valid_steps, 5, 3] -> view(-1, 3)
            # target_disc:       [B, K, 5]    -> mask select -> [valid_steps, 5]    -> view(-1)

            valid_logits = preds_disc_logits[mask.bool()]  # [Valid, 5, 3]
            valid_targets = target_disc[mask.bool()]  # [Valid, 5]

            loss_d = loss_fn_disc(valid_logits.view(-1, 3), valid_targets.view(-1))

            # Total Loss (权重 1:1)
            loss = loss_c + loss_d

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            losses.append(loss.item())
            pbar.set_postfix({'L_Cont': f"{loss_c.item():.2f}", 'L_Disc': f"{loss_d.item():.2f}"})

        scheduler.step()
        avg_loss = np.mean(losses)
        print(f"Epoch {epoch + 1} done. Avg Loss: {avg_loss:.4f}")

        # Save Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_dir / "best_dt_baseline.pth")

        # Closed-Loop Eval (Optional)
        # if (epoch + 1) % 5 == 0:
        #     # 需要实例化一个 Ray Env (开销较大，且需配置正确)
        #     pass

    print(f"✅ Baseline DT Training Finished. Saved to {save_dir}")


if __name__ == "__main__":
    # 示例入口 (实际应由 Hydra 启动)
    pass