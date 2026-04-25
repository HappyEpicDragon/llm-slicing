"""
PPO-Lagrangian Baseline 训练入口。

使用与 ppo-baseline / dt-baseline **相同的仿真环境**（CommunicationEnv + IBSched），
在 s0–s4 的 ep 0–59 上训练，seed=0。

与 ppo_lagrangian_v2（EnvV2）的唯一区别：
  - 环境后端换成 CommunicationEnv（env_sb3.py）
  - 观测：IBSched 格式的 145 维扁平向量
  - 动作：Box(10)  → 前5维连续 inter-slice 得分，后5维连续 intra-slice logit（argmax→0/1/2）

用法（Hydra）：
    pixi run sim channel_generality/train_ppo_lagrangian_baseline
"""
import os
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig, OmegaConf

from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import LagrangianPPO
from src.basic_apis.ppo.ppo_lagrangian.obs_dim_utils import LAGRANGIAN_BASELINE_OBS_DIM
from src.basic_apis.asset_utils import (
    build_versioned_run_dir, update_latest_symlink, ensure_clean_dir, ensure_dir
)

NUM_SLICES = 5


def _build_comm_env_cfg(base_env_cfg: dict, path_context,
                        active_scenarios: list, seed: int) -> dict:
    """从 single_scenario_env 格式的 Hydra 环境配置构造 CommunicationEnv 所需的 dict。"""
    from hydra.utils import get_class

    cfg = dict(base_env_cfg)
    cfg['path_context'] = path_context
    cfg['seed'] = seed
    cfg['seed_test'] = seed
    cfg['mode'] = 'training'
    scenario_mode = cfg.get('scenario_mode', 'inside')
    cfg['scenario_mode'] = scenario_mode
    cfg['model_name'] = cfg.get('model_name', 'lagrangian_baseline')

    # 设置训练场景列表
    cfg[scenario_mode]['training']['active_scenario_list'] = active_scenarios

    # 把字符串类路径转换为 class 对象
    for field in ['channel_class', 'association_class']:
        val = cfg.get(field, {})
        if isinstance(val, str):
            cfg[field] = get_class(val)
        elif isinstance(val, dict):
            cfg[field] = get_class(val[scenario_mode])
    for field in ['traffic_class', 'mobility_class']:
        val = cfg.get(field, '')
        if isinstance(val, str) and val:
            cfg[field] = get_class(val)

    # 初始化 state_config（CommunicationEnv.__init__ 还会覆写一遍，但要先有 key）
    sm_cfg = cfg[scenario_mode]
    state_cfg = cfg.get('state_config', {})
    if isinstance(state_cfg, dict):
        state_cfg = dict(state_cfg)
    state_cfg['scenario_skip_episodes'] = sm_cfg.get('scenario_skip_episodes', 100)
    state_cfg['initial_episode'] = sm_cfg['training']['initial_episode']
    state_cfg['max_episode'] = sm_cfg['training']['max_episode']
    state_cfg['active_scenario_list'] = active_scenarios
    state_cfg.setdefault('max_number_steps', 1000)
    cfg['state_config'] = state_cfg

    return cfg


class LagrangianBaselineEnv(gym.Env):
    """
    SB3 单智能体 Gym 包装器，底层使用 CommunicationEnv + IBSched。

    - Obs : flat Box(145)  ← IBSched.obs_space_format 的所有 player 观测拼接
    - Action : Box(10)     ← 前 5 维 inter-slice 连续得分；后 5 维 intra-slice logit
    - Reward : IBSched 计算的所有 player reward 之和（标量）
    - Info : CommunicationEnv._build_drift_info() 的 drift/meta 字段
              （用于 Lagrangian cost 和评测指标）
    """

    OBS_DIM = LAGRANGIAN_BASELINE_OBS_DIM  # 145

    def __init__(self, comm_env_cfg: dict):
        super().__init__()

        from src.basic_apis.ppo.ppo_baseline.env_sb3 import CommunicationEnv
        from src.basic_apis.ppo.ppo_baseline.env_ray import MARLCommEnv
        from src.basic_apis.ppo.ppo_baseline.agent_ib_sched import IBSched

        self._comm_env = CommunicationEnv(comm_env_cfg)
        comps = self._comm_env.components

        # 必须先建 MARLCommEnv（IBSched 的构造函数会 assert isinstance(env, MARLCommEnv)）
        self._marl_env = MARLCommEnv.__new__(MARLCommEnv)
        self._marl_env.comm_env = self._comm_env
        num_slices_p1 = comps.slices.max_number_slices + 1
        self._marl_env.agents = {f"player_{r}" for r in range(num_slices_p1)}
        self._marl_env._agent_ids = set(self._marl_env.agents)
        # 让 isinstance 检查通过
        from gymnasium import spaces as gym_spaces
        self._marl_env.observation_space = gym_spaces.Space()
        self._marl_env.action_space = gym_spaces.Space()

        num_ues = self._comm_env.config.ue_config.max_number_ues
        num_slices = self._comm_env.config.slice_config.max_number_slices
        num_bs = self._comm_env.config.basestation_config.max_number_basestations
        num_rbs = self._comm_env.config.basestation_config.num_available_rbs

        self._agent = IBSched(
            env=self._marl_env,
            max_number_ues=num_ues,
            max_number_slices=num_slices,
            max_number_basestations=num_bs,
            num_available_rbs=num_rbs,
        )

        # 注入 obs/reward/action 函数
        self._comm_env.set_agent_functions(
            obs_space_format=self._agent.obs_space_format,
            action_format=self._agent.action_format,
            calculate_reward=self._agent.calculate_reward,
            obs_space=self._agent.get_obs_space(),
            action_space=self._agent.get_action_space(),
        )

        self._num_slices = num_slices

        # SB3 兼容的扁平空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # 前 num_slices 维：inter-slice 连续得分；后 num_slices 维：intra-slice logit
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(num_slices * 2,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs_dict, info = self._comm_env.reset(seed=seed, options=options)
        return self._flatten_obs(obs_dict), info

    def step(self, action: np.ndarray):
        action_dict = self._build_marl_action(action)
        obs_dict, reward_raw, done, truncated, info = self._comm_env.step(action_dict)
        reward = self._aggregate_reward(reward_raw)
        flat_obs = self._flatten_obs(obs_dict)
        return flat_obs, reward, done, truncated, info

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        """将 IBSched 返回的 player_* obs dict 展平为 145 维向量。"""
        parts = []
        for key in ['player_0'] + [f'player_{i}' for i in range(1, self._num_slices + 1)]:
            if key not in obs_dict:
                continue
            val = obs_dict[key]
            arr = val.get('observations', np.array([])) if isinstance(val, dict) else val
            parts.append(np.asarray(arr, dtype=np.float32).flatten())
        flat = np.concatenate(parts) if parts else np.zeros(self.OBS_DIM, dtype=np.float32)
        # 对齐到固定长度
        if len(flat) >= self.OBS_DIM:
            return flat[:self.OBS_DIM]
        return np.pad(flat, (0, self.OBS_DIM - len(flat))).astype(np.float32)

    def _build_marl_action(self, action: np.ndarray) -> dict:
        """Box(10) → MARL action dict: {player_0: Box(5), player_1-5: int}"""
        n = self._num_slices
        inter_scores = np.asarray(action[:n], dtype=np.float64)
        intra_logits = np.asarray(action[n:], dtype=np.float64)
        # 将 intra logit ([-1,1]) 线性映射到 {0,1,2}
        intra_choices = np.clip(
            np.floor((intra_logits + 1.0) / 2.0 * 3.0).astype(int), 0, 2
        )
        action_dict = {"player_0": inter_scores}
        for i in range(n):
            action_dict[f"player_{i + 1}"] = int(intra_choices[i])
        return action_dict

    @staticmethod
    def _aggregate_reward(reward_raw) -> float:
        """将 per-player reward dict 聚合为标量。"""
        if isinstance(reward_raw, dict):
            vals = [float(v) for v in reward_raw.values() if isinstance(v, (int, float))]
            return float(np.sum(vals)) if vals else 0.0
        return float(reward_raw)


class LagrangianAugmentedBaselineEnv(gym.Env):
    """
    在 LagrangianBaselineEnv 外层再包一层 Lagrangian 奖励增广。
    供 SB3 DummyVecEnv 使用。
    """

    def __init__(self, comm_env_cfg: dict, lagrangian: "LagrangianPPO"):
        super().__init__()
        self._inner = LagrangianBaselineEnv(comm_env_cfg)
        self.lagrangian = lagrangian
        self.observation_space = self._inner.observation_space
        self.action_space = self._inner.action_space

    def reset(self, **kwargs):
        return self._inner.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._inner.step(action)
        aug_reward = self.lagrangian.compute_augmented_reward(
            reward, self.lagrangian.compute_cost(info)
        )
        return obs, aug_reward, terminated, truncated, info

    def close(self):
        self._inner.close()


def train_ppo_lagrangian_baseline(cfg: DictConfig, path_context):
    """PPO-Lagrangian Baseline 训练主函数（使用 CommunicationEnv + IBSched 环境）。"""
    train_cfg = cfg.train_ppo_lagrangian_baseline

    seed = int(train_cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 训练场景
    scenarios = list(train_cfg.get('scenarios', [0, 1, 2, 3, 4]))

    # 从 Hydra 环境配置构建 CommunicationEnv cfg dict
    base_env_cfg = OmegaConf.to_container(cfg.environment, resolve=True)
    comm_env_cfg = _build_comm_env_cfg(base_env_cfg, path_context, scenarios, seed)

    # 保存路径
    save_path = str(train_cfg.get(
        'save_path',
        f"data/channel_generality/ppo_lagrangian_baseline/models/"
        f"scen{'_'.join(map(str, scenarios))}_seed{seed}"
    ))
    os.makedirs(os.path.dirname(os.path.abspath(save_path + "_ppo.zip")), exist_ok=True)

    asset_cfg = train_cfg.get('asset', None)
    run_dir = None
    if asset_cfg and bool(asset_cfg.get('use_versioned_runs', False)):
        run_id_cfg = asset_cfg.get('run_id', 'auto')
        run_id = None if str(run_id_cfg) == 'auto' else str(run_id_cfg)
        run_dir = build_versioned_run_dir(save_path, run_id=run_id)
        if bool(asset_cfg.get('clean_before_run', False)):
            ensure_clean_dir(run_dir)
        else:
            ensure_dir(run_dir)
        save_path = str(Path(run_dir) / "model")
        print(f"[Asset] versioned run dir: {run_dir}")

    # 构建 Lagrangian 对象
    from stable_baselines3.common.vec_env import DummyVecEnv
    from src.basic_apis.ppo.ppo_lagrangian.cost_network import CostNetwork
    from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import (
        LagrangianCallback, LagrangianRewardWrapper
    )

    device = str(train_cfg.get('device', 'cpu'))
    lagrangian = LagrangianPPO.__new__(LagrangianPPO)
    lagrangian.cost_limit = float(train_cfg.get('cost_limit', 1e-4))
    lagrangian.lagrangian_lr = float(train_cfg.get('lagrangian_lr', 1e-7))
    lagrangian.lagrangian_multiplier = float(train_cfg.get('initial_lambda', 10.0))
    lagrangian.cost_critic = CostNetwork(LAGRANGIAN_BASELINE_OBS_DIM, hidden=128).to(device)
    lagrangian.cost_optimizer = torch.optim.Adam(
        lagrangian.cost_critic.parameters(),
        lr=float(train_cfg.get('critic_lr', 1e-3))
    )
    lagrangian.device = device
    lagrangian.reward_wrapper = LagrangianRewardWrapper(lagrangian)

    def make_env_fn(cfg_dict):
        def _init():
            return LagrangianAugmentedBaselineEnv(cfg_dict, lagrangian)
        return _init

    aug_vec_env = DummyVecEnv([make_env_fn(comm_env_cfg)])

    from stable_baselines3 import PPO
    lagrangian.ppo = PPO(
        policy="MlpPolicy",
        env=aug_vec_env,
        learning_rate=float(train_cfg.get('actor_lr', 8e-5)),
        n_steps=int(train_cfg.get('n_steps_per_rollout', 500)),
        batch_size=64,
        gamma=float(train_cfg.get('gamma', 0.4)),
        verbose=1,
        device=device,
    )
    lagrangian._lagrangian_cb = LagrangianCallback(lagrangian, verbose=1)

    total_timesteps = int(train_cfg.get('total_timesteps', 1_000_000))
    print(f"Training PPO-Lagrangian Baseline: scenarios={scenarios}, seed={seed}, "
          f"total_steps={total_timesteps}")

    lagrangian.learn(total_timesteps=total_timesteps)
    lagrangian.save(save_path)
    print(f"Model saved to: {save_path}")

    if asset_cfg and bool(asset_cfg.get('use_versioned_runs', False)) and bool(asset_cfg.get('update_latest', True)):
        if run_dir is None:
            run_dir = str(Path(save_path).parent)
        link_path = update_latest_symlink(
            str(train_cfg.get('save_path')), run_dir
        )
        print(f"[Asset] updated latest symlink: {link_path}")

    aug_vec_env.close()
