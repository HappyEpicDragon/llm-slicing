"""
Lagrangian PPO 训练入口（B2 Baseline）。

用法：
    # 单场景训练
    python main.py name=channel_generality mode=train_ppo_lagrangian \
        train_ppo_lagrangian.scenario=0

    # 跨场景联合训练
    python main.py name=channel_generality mode=train_ppo_lagrangian \
        "train_ppo_lagrangian.scenarios=[0,1,2,3,4]"
"""
import os
import random
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.basic_apis.dt.env import HierarchicalSlicingEnvV2
from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import LagrangianPPO
from src.basic_apis.ppo.ppo_lagrangian.obs_dim_utils import lagrangian_flat_obs_dim_v2
from src.basic_apis.utils.assets import build_versioned_run_dir, update_latest_symlink, ensure_clean_dir, ensure_dir


class LagrangianAugmentedEnv(gym.Env):
    """
    Gym 包装层：在 step() 中将原始 reward 替换为 Lagrangian augmented reward。
    lagrangian 对象在训练过程中持续更新 λ，包装层每步都读取最新的 λ。
    """

    def __init__(self, env_settings, np_random, paths_cfg, workdir, lagrangian: "LagrangianPPO"):
        super().__init__()
        self._env = HierarchicalSlicingEnvV2(
            env_settings,
            np_random,
            paths_cfg=paths_cfg,
            workdir=workdir,
        )
        self.lagrangian = lagrangian
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        aug_reward = self.lagrangian.compute_augmented_reward(
            float(reward), self.lagrangian.compute_cost(info)
        )
        return obs, aug_reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def render(self, mode="human"):
        return self._env.render(mode)


def train_ppo_lagrangian(cfg: DictConfig, paths_cfg=None, workdir=None):
    """Lagrangian PPO 训练主函数"""
    train_cfg = cfg.train_ppo_lagrangian
    paths_cfg = paths_cfg if paths_cfg is not None else cfg.get("paths", None)
    workdir = workdir if workdir is not None else str(cfg.get("workdir", os.getcwd()))

    seed = int(train_cfg.get('seed', 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 确定训练场景：scenarios（多场景）优先于 scenario（单场景）
    scenario = train_cfg.get('scenario', None)
    scenarios = train_cfg.get('scenarios', None)
    if scenarios is not None:
        active_scenarios = list(scenarios)
    elif scenario is not None:
        active_scenarios = [int(scenario)]
    else:
        active_scenarios = [0]

    # 构建环境配置（转为普通 dict 以便 DummyVecEnv 工厂函数 pickle）
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv

    env_settings = OmegaConf.to_container(cfg.environment.env_settings, resolve=True)
    env_settings['mode'] = 'training'
    scenario_mode = env_settings.get('scenario_mode', 'inside')
    env_settings[scenario_mode]['training']['active_scenario_list'] = active_scenarios

    # CostNetwork 扁平输入维（EnvV2：与 inter/intra/global 展平一致；旧 V1 为 45）
    obs_dim = lagrangian_flat_obs_dim_v2(env_settings)

    total_timesteps = int(train_cfg.get('total_timesteps', 1_000_000))
    save_path = str(train_cfg.get('save_path',
                                  f"data/channel_generality/ppo_lagrangian/models/"
                                  f"scen{'_'.join(map(str, active_scenarios))}_seed{seed}"))
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
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

    # Step 1: 先创建一个临时占位 env 来初始化 LagrangianPPO（获取 obs/action space）
    # Step 2: 用包含 Lagrangian 奖励增广的 env 替换 PPO 内部的 env
    from omegaconf import OmegaConf as OC
    env_settings_node = OC.create(env_settings)

    # 创建 LagrangianPPO，使用 LagrangianAugmentedEnv 包装器
    # 需要先创建 lagrangian 对象（用于闭包引用），然后再建包装 env
    lagrangian = LagrangianPPO.__new__(LagrangianPPO)
    lagrangian.cost_limit = float(train_cfg.get('cost_limit', 1e-4))
    lagrangian.lagrangian_lr = float(train_cfg.get('lagrangian_lr', 1e-7))
    lagrangian.lagrangian_multiplier = float(train_cfg.get('initial_lambda', 10.0))
    from src.basic_apis.ppo.ppo_lagrangian.cost_network import CostNetwork
    device = str(train_cfg.get('device', 'cpu'))
    lagrangian.cost_critic = CostNetwork(obs_dim, hidden=128).to(device)
    lagrangian.cost_optimizer = torch.optim.Adam(
        lagrangian.cost_critic.parameters(),
        lr=float(train_cfg.get('critic_lr', 1e-3))
    )
    lagrangian.device = device
    from src.basic_apis.ppo.ppo_lagrangian.lagrangian_ppo import LagrangianRewardWrapper, LagrangianCallback
    lagrangian.reward_wrapper = LagrangianRewardWrapper(lagrangian)

    def make_aug_env_fn(env_s, rng_seed):
        def _init():
            return LagrangianAugmentedEnv(
                OC.create(env_s),
                np.random.default_rng(rng_seed),
                paths_cfg,
                workdir,
                lagrangian,
            )
        return _init

    aug_vec_env = DummyVecEnv([make_aug_env_fn(env_settings, seed)])

    from stable_baselines3 import PPO
    lagrangian.ppo = PPO(
        policy="MultiInputPolicy",
        env=aug_vec_env,
        learning_rate=float(train_cfg.get('actor_lr', 8e-5)),
        n_steps=int(train_cfg.get('n_steps_per_rollout', 500)),
        batch_size=64,
        gamma=float(train_cfg.get('gamma', 0.4)),
        verbose=1,
        device=device,
    )
    lagrangian._lagrangian_cb = LagrangianCallback(lagrangian, verbose=1)

    print(f"Training Lagrangian PPO: scenarios={active_scenarios}, seed={seed}, "
          f"total_steps={total_timesteps}")
    lagrangian.learn(total_timesteps=total_timesteps)
    lagrangian.save(save_path)
    print(f"Model saved to: {save_path}")

    if asset_cfg and bool(asset_cfg.get('use_versioned_runs', False)) and bool(asset_cfg.get('update_latest', True)):
        if run_dir is None:
            run_dir = str(Path(save_path).parent)
        link_path = update_latest_symlink(str(train_cfg.get('save_path')), run_dir)
        print(f"[Asset] updated latest symlink: {link_path}")

    aug_vec_env.close()
