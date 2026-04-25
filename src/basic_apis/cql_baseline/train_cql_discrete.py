"""
CQL-v3: 多头离散 Q 网络，直接在 MultiDiscrete 动作空间上训练。

与 CQL-v2 的唯一区别：
  - CQL-v2: 把 11-dim 离散动作转换为 5-dim 连续 PRB 分配比例后再学习
  - CQL-v3: 直接对原始 11-dim 离散动作学习 Q 函数（独立多头）

动作空间：[codebook(11), ordering×5(各5), intra×5(各3)] = 11 维
观测空间：45-dim 扁平（与 CQL-v2 一致）

架构：共享编码器 + 11 个独立 Q-head，每个 head 对应一个动作维度。
训练：独立 CQL loss（每个 head 独立 Bellman + logsumexp 保守惩罚），共享奖励。
"""

import os
import json
import random
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional


# ─── 常量 ───────────────────────────────────────────────────────────────────

# MultiDiscrete 动作维度：[codebook(11), 5×ordering(5), 5×intra(3)]
ACTION_DIMS: List[int] = [11] + [5] * 5 + [3] * 5
NUM_ACTION_HEADS: int = len(ACTION_DIMS)  # 11
OBS_DIM: int = 45
NUM_SLICES: int = 5
USERS_PER_SLICE: int = 5


# ─── 观测转换（与 CQL-v2 保持一致）──────────────────────────────────────────

def convert_obs_v2_to_flat45(obs_dict) -> np.ndarray:
    """V2 结构化观测 → 45-dim 扁平向量（与 convert_dt_dataset_to_cql.py 保持一致）。"""
    inter = np.asarray(obs_dict["inter_feat"], dtype=np.float32)   # (5, 8)
    intra = np.asarray(obs_dict["intra_feat"], dtype=np.float32)   # (25, 7)
    inter_v1 = inter[:, :4]                                         # (5, 4)
    intra_v1 = intra[:, :5]                                         # (25, 5)
    intra_per_slice = intra_v1.reshape(NUM_SLICES, USERS_PER_SLICE, 5).mean(axis=1)  # (5, 5)
    return np.concatenate([inter_v1.flatten(), intra_per_slice.flatten()])  # (45,)


# ─── 模型 ────────────────────────────────────────────────────────────────────

class MultiHeadQNetwork(nn.Module):
    """共享 MLP 编码器 + 11 个独立 Q-head（每个对应一个动作维度）。"""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 256,
                 action_dims: List[int] = ACTION_DIMS):
        super().__init__()
        self.action_dims = action_dims
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n) for n in action_dims
        ])

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        """返回长度 11 的列表，每个元素形状 (B, n_actions_i)。"""
        h = self.encoder(obs)
        return [head(h) for head in self.heads]


# ─── 数据集 ──────────────────────────────────────────────────────────────────

class TransitionBuffer:
    """从 D-B pkl 文件加载轨迹并存储成 transition 数组。"""

    def __init__(self, dataset_dir: str, max_trajs: Optional[int] = None):
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.next_obs: List[np.ndarray] = []
        self.dones: List[float] = []
        self._load(dataset_dir, max_trajs)

    def _load(self, dataset_dir: str, max_trajs: Optional[int]):
        pkl_files = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".pkl"))
        if max_trajs is not None:
            pkl_files = pkl_files[:max_trajs]
        print(f"[CQL-v3] Loading {len(pkl_files)} trajectories from {dataset_dir}")
        for fname in pkl_files:
            data = np.load(os.path.join(dataset_dir, fname), allow_pickle=True)
            obs_list = data["observations"]     # list of dicts
            acts = np.array(data["actions"], dtype=np.int64)   # (T, 11)
            rwds = np.array(data["rewards"], dtype=np.float32) # (T,)
            dones = np.array(data["dones"], dtype=np.float32)  # (T,)
            T = len(obs_list)
            for t in range(T - 1):
                self.obs.append(convert_obs_v2_to_flat45(obs_list[t]))
                self.actions.append(acts[t])
                self.rewards.append(float(rwds[t]))
                self.next_obs.append(convert_obs_v2_to_flat45(obs_list[t + 1]))
                self.dones.append(float(dones[t]))
            # 最后一步：next_obs 用 obs[T-1] 本身，done=1
            self.obs.append(convert_obs_v2_to_flat45(obs_list[T - 1]))
            self.actions.append(acts[T - 1])
            self.rewards.append(float(rwds[T - 1]))
            self.next_obs.append(convert_obs_v2_to_flat45(obs_list[T - 1]))
            self.dones.append(1.0)

        self.obs = np.stack(self.obs, axis=0).astype(np.float32)
        self.actions = np.stack(self.actions, axis=0).astype(np.int64)
        self.rewards = np.array(self.rewards, dtype=np.float32)
        self.next_obs = np.stack(self.next_obs, axis=0).astype(np.float32)
        self.dones = np.array(self.dones, dtype=np.float32)
        print(f"[CQL-v3] Dataset: {len(self.obs)} transitions")

    def sample(self, batch_size: int):
        idx = np.random.randint(0, len(self.obs), size=batch_size)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return len(self.obs)


# ─── 训练 ────────────────────────────────────────────────────────────────────

def train_cql_discrete(
    dataset_dir: str,
    output_dir: str,
    seed: int = 0,
    n_steps: int = 100_000,
    batch_size: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    alpha: float = 1.0,           # CQL 保守系数
    target_update_freq: int = 200,
    hidden_dim: int = 256,
    device: str = "cuda:0",
    max_trajs: Optional[int] = None,
):
    """训练多头离散 CQL（CQL-v3）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # 数据集
    buffer = TransitionBuffer(dataset_dir, max_trajs=max_trajs)

    # 网络
    q_net = MultiHeadQNetwork(obs_dim=OBS_DIM, hidden_dim=hidden_dim).to(device)
    q_target = MultiHeadQNetwork(obs_dim=OBS_DIM, hidden_dim=hidden_dim).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

    log = {"td_loss": [], "cql_loss": [], "total_loss": []}

    print(f"[CQL-v3] Training: seed={seed}, steps={n_steps}, "
          f"batch={batch_size}, alpha={alpha}, device={device}")

    for step in range(1, n_steps + 1):
        obs_b, act_b, rew_b, next_obs_b, done_b = buffer.sample(batch_size)
        obs_t      = torch.tensor(obs_b,      device=device)
        act_t      = torch.tensor(act_b,      device=device)       # (B, 11) int64
        rew_t      = torch.tensor(rew_b,      device=device)       # (B,)
        next_obs_t = torch.tensor(next_obs_b, device=device)
        done_t     = torch.tensor(done_b,     device=device)       # (B,)

        # ── Q 值（当前状态）──
        q_preds = q_net(obs_t)           # list of (B, n_actions_i)

        # ── 目标 Q 值（下一状态，贪心策略）──
        with torch.no_grad():
            q_next_preds = q_target(next_obs_t)

        td_loss_sum = 0.0
        cql_loss_sum = 0.0
        for i, (q_pred_i, q_next_i) in enumerate(zip(q_preds, q_next_preds)):
            a_i = act_t[:, i]                               # (B,) int64
            q_a = q_pred_i.gather(1, a_i.unsqueeze(1)).squeeze(1)  # (B,)

            # Bellman target
            target = rew_t + gamma * q_next_i.max(dim=-1).values * (1.0 - done_t)
            td_loss_sum += F.mse_loss(q_a, target.detach())

            # CQL 保守惩罚：logsumexp(Q) - Q(a_data)
            cql_loss_sum += (q_pred_i.logsumexp(dim=-1) - q_a).mean()

        total_loss = td_loss_sum + alpha * cql_loss_sum

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()

        # 软更新目标网络
        if step % target_update_freq == 0:
            q_target.load_state_dict(q_net.state_dict())

        if step % 1000 == 0 or step == n_steps:
            log["td_loss"].append(float(td_loss_sum.item()))
            log["cql_loss"].append(float(cql_loss_sum.item()))
            log["total_loss"].append(float(total_loss.item()))
            print(f"  step {step:6d}/{n_steps}  "
                  f"td={td_loss_sum.item():.4f}  "
                  f"cql={cql_loss_sum.item():.4f}  "
                  f"total={total_loss.item():.4f}")

    # 保存模型
    model_path = os.path.join(output_dir, "model.pth")
    torch.save({
        "q_net_state_dict": q_net.state_dict(),
        "action_dims": ACTION_DIMS,
        "obs_dim": OBS_DIM,
        "hidden_dim": hidden_dim,
        "seed": seed,
    }, model_path)
    print(f"[CQL-v3] Model saved: {model_path}")

    # 保存训练曲线
    with open(os.path.join(output_dir, "train_log.json"), "w") as f:
        json.dump(log, f)

    return model_path


# ─── 测试 ─────────────────────────────────────────────────────────────────────

def _load_q_net(model_path: str, device: torch.device) -> MultiHeadQNetwork:
    ckpt = torch.load(model_path, map_location=device)
    obs_dim   = ckpt.get("obs_dim", OBS_DIM)
    hidden_dim = ckpt.get("hidden_dim", 256)
    action_dims = ckpt.get("action_dims", ACTION_DIMS)
    q_net = MultiHeadQNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim,
                               action_dims=action_dims).to(device)
    q_net.load_state_dict(ckpt["q_net_state_dict"])
    q_net.eval()
    return q_net


def _select_action(q_net: MultiHeadQNetwork, obs_flat: np.ndarray,
                   device: torch.device) -> np.ndarray:
    with torch.no_grad():
        obs_t = torch.tensor(obs_flat[np.newaxis], dtype=torch.float32, device=device)
        q_preds = q_net(obs_t)          # list of (1, n_actions_i)
        action = np.array([q.argmax(dim=-1).item() for q in q_preds], dtype=np.int64)
    return action


def _compute_step_metrics(info, num_slices=5):
    hp_dist = nhp_dist = 0.0
    hp_viols = nhp_viols = hp_active = nhp_active = 0
    for s_idx in range(num_slices):
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


def test_cql_discrete_scenario(
    model_path: str,
    env_config: dict,
    scenario_id: int,
    seed: int,
    n_episodes: int = 100,
    save_root: str = "data/channel_generality/cql_v3",
    device_str: str = "cuda:0",
):
    """在单个场景+seed 上评测 CQL-v3，保存 summary.json + metric JSON。"""
    from omegaconf import OmegaConf
    from src.basic_apis.network_slicing_business.path_context import PathContext
    from src.basic_apis.dt_v2.env_v2 import HierarchicalSlicingEnvV2

    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    q_net = _load_q_net(model_path, device)
    print(f"[CQL-v3] Loaded model: {model_path}")

    env_settings = dict(env_config.get("env_settings", env_config))
    scenario_mode = env_settings.get("scenario_mode", "inside")
    env_settings["mode"] = "testing"
    env_settings[scenario_mode]["testing"]["active_scenario_list"] = [scenario_id]

    cfg_node = OmegaConf.create(env_settings)
    path_context = PathContext("./outputs/cql_v3_test")
    env = HierarchicalSlicingEnvV2(cfg_node, np.random.default_rng(seed), path_context)

    ep_rewards, ep_hp_viols, ep_nhp_viols = [], [], []
    ep_hp_dists, ep_nhp_dists = [], []
    all_step_hp_dists, all_step_nhp_dists = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        ep_hp_dist_sum = ep_nhp_dist_sum = 0.0
        ep_hp_viol_sum = ep_nhp_viol_sum = 0
        ep_hp_active = ep_nhp_active = 0

        step_hp_dists, step_nhp_dists = [], []
        while not done:
            flat_obs = convert_obs_v2_to_flat45(obs)
            action = _select_action(q_net, flat_obs, device)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            hp_d, nhp_d, hp_v, nhp_v, hp_a, nhp_a = _compute_step_metrics(info)
            ep_hp_dist_sum += hp_d
            ep_nhp_dist_sum += nhp_d
            ep_hp_viol_sum += hp_v
            ep_nhp_viol_sum += nhp_v
            ep_hp_active += hp_a
            ep_nhp_active += nhp_a
            step_hp_dists.append(hp_d / hp_a if hp_a > 0 else 0.0)
            step_nhp_dists.append(nhp_d / nhp_a if nhp_a > 0 else 0.0)

        hp_dist_ep  = ep_hp_dist_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
        nhp_dist_ep = ep_nhp_dist_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0
        hp_viol_ep  = ep_hp_viol_sum  / ep_hp_active  if ep_hp_active  > 0 else 0.0
        nhp_viol_ep = ep_nhp_viol_sum / ep_nhp_active if ep_nhp_active > 0 else 0.0

        ep_rewards.append(ep_reward)
        ep_hp_viols.append(float(hp_viol_ep))
        ep_nhp_viols.append(float(nhp_viol_ep))
        ep_hp_dists.append(float(hp_dist_ep))
        ep_nhp_dists.append(float(nhp_dist_ep))
        all_step_hp_dists.append(step_hp_dists)
        all_step_nhp_dists.append(step_nhp_dists)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  s{scenario_id} seed{seed} ep{ep+1}: "
                  f"rwd={ep_reward:.2f}  hp_v={hp_viol_ep:.4f}  nhp_v={nhp_viol_ep:.4f}")

    env.close()

    summary = {
        "scenario_id": scenario_id,
        "seed": seed,
        "n_episodes": n_episodes,
        "reward_mean": float(np.mean(ep_rewards)),
        "reward_std":  float(np.std(ep_rewards)),
        "hp_viol_mean":  float(np.mean(ep_hp_viols)),
        "nhp_viol_mean": float(np.mean(ep_nhp_viols)),
        "hp_dist_mean":  float(np.mean(ep_hp_dists)),
        "nhp_dist_mean": float(np.mean(ep_nhp_dists)),
        "combined": float(np.mean(ep_nhp_viols)),
    }

    # 保存路径：metric_json/scenario_X/seed_Y/summary.json
    save_dir = os.path.join(save_root, "cql_expert", "metric_json",
                            f"scenario_{scenario_id}", f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # 保存各项指标的 JSON（对齐 dt_v2/test_v2.py 格式）
    for key, vals in [("hp_violations", ep_hp_viols), ("nhp_violations", ep_nhp_viols),
                      ("hp_distance", ep_hp_dists), ("nhp_distance", ep_nhp_dists)]:
        with open(os.path.join(save_dir, f"{key}.json"), "w") as f:
            json.dump({key: vals, "mean": float(np.mean(vals)), "episodes": n_episodes}, f, indent=2)

    raw_dir = os.path.join(save_root, "cql_expert", "metric_raw", f"scenario_{scenario_id}")
    os.makedirs(raw_dir, exist_ok=True)
    max_steps = max((len(s) for s in all_step_hp_dists), default=0)

    def _pad_steps(step_list, target_len):
        arr = np.array(step_list, dtype=np.float64)
        if len(arr) < target_len:
            arr = np.concatenate([arr, np.zeros(target_len - len(arr), dtype=np.float64)])
        return arr

    np.savez(
        os.path.join(raw_dir, f"ep_seed{seed}.npz"),
        ep_rewards=np.array(ep_rewards, dtype=np.float64),
        hp_viols=np.array(ep_hp_viols, dtype=np.float64),
        nhp_viols=np.array(ep_nhp_viols, dtype=np.float64),
        step_hp_dist=np.array([_pad_steps(s, max_steps) for s in all_step_hp_dists], dtype=np.float64),
        step_nhp_dist=np.array([_pad_steps(s, max_steps) for s in all_step_nhp_dists], dtype=np.float64),
    )

    print(f"[CQL-v3] Saved metrics → {save_dir}")
    return summary


# ─── Hydra 入口 ──────────────────────────────────────────────────────────────

def train(cfg, path_context):
    """train_cql_v3 模式的 Hydra 入口。"""
    tc = cfg.train_cql_v3
    seeds = list(tc.get("train_seeds", [0, 1, 2, 3, 4]))
    dataset_dir = str(tc.dataset_dir)
    output_root = str(tc.output_root)

    for seed in seeds:
        output_dir = os.path.join(output_root, f"expert_seed{seed}")
        train_cql_discrete(
            dataset_dir  = dataset_dir,
            output_dir   = output_dir,
            seed         = seed,
            n_steps      = int(tc.get("n_steps", 100_000)),
            batch_size   = int(tc.get("batch_size", 256)),
            lr           = float(tc.get("lr", 3e-4)),
            gamma        = float(tc.get("gamma", 0.99)),
            alpha        = float(tc.get("alpha", 1.0)),
            hidden_dim   = int(tc.get("hidden_dim", 256)),
            target_update_freq = int(tc.get("target_update_freq", 200)),
            device       = str(tc.get("device", "cuda:0")),
        )


def test(cfg, path_context):
    """test_cql_v3 模式的 Hydra 入口。"""
    from omegaconf import OmegaConf
    tc = cfg.test_cql_v3
    env_config = OmegaConf.to_container(cfg.environment, resolve=False)

    model_root  = str(tc.get("model_root",  "data/channel_generality/cql_v3/models"))
    save_root   = str(tc.get("save_root",   "data/channel_generality/cql_v3"))
    seeds       = list(tc.get("test_seeds",    [0, 1, 2, 3, 4]))
    scenarios   = list(tc.get("test_scenarios",[5, 6, 7, 8, 9]))
    n_episodes  = int(tc.get("n_episodes",  100))
    device_str  = str(tc.get("device", "cuda:0"))
    model_seed_cfg = tc.get("model_seed", None)

    for seed in seeds:
        model_seed = int(model_seed_cfg) if model_seed_cfg is not None else seed
        model_path = os.path.join(model_root, f"expert_seed{model_seed}", "model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CQL-v3 model not found: {model_path}")
        for scen in scenarios:
            test_cql_discrete_scenario(
                model_path   = model_path,
                env_config   = env_config,
                scenario_id  = scen,
                seed         = seed,
                n_episodes   = n_episodes,
                save_root    = save_root,
                device_str   = device_str,
            )
