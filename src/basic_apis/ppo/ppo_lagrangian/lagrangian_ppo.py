"""
Lagrangian PPO — Constrained RL Baseline [20]。

在 Stable Baselines3 PPO 基础上增加 Lagrangian 约束机制：
  augmented_reward = r - λ * cost
  λ 按对偶梯度上升更新：λ ← ReLU(λ + lr_λ * (mean_cost - cost_limit))

参考文献：
  M. Zangooei et al., "Flexible RAN Slicing in Open RAN With Constrained
  Multi-Agent Reinforcement Learning," IEEE JSAC, 2024. [20]
"""
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.basic_apis.ppo.ppo_lagrangian.cost_network import CostNetwork


class LagrangianCallback(BaseCallback):
    """
    在每个 rollout 结束后更新 Lagrangian multiplier λ。
    同时收集 episode cost 并写入 log。
    """

    def __init__(self, lagrangian: "LagrangianPPO", verbose: int = 0):
        super().__init__(verbose)
        self.lagrangian = lagrangian
        self._episode_costs: list = []
        self._current_ep_cost: float = 0.0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            cost = self.lagrangian.compute_cost(info)
            self._current_ep_cost += cost

            if dones[i]:
                self._episode_costs.append(self._current_ep_cost)
                self._current_ep_cost = 0.0

        return True

    def _on_rollout_end(self) -> None:
        if self._episode_costs:
            mean_cost = float(np.mean(self._episode_costs))
            self.lagrangian.update_lagrangian(mean_cost)
            if self.verbose >= 1:
                print(f"[Lagrangian] λ={self.lagrangian.lagrangian_multiplier:.4f}, "
                      f"mean_cost={mean_cost:.6f}")
            self._episode_costs = []


class LagrangianRewardWrapper:
    """
    对环境 reward 进行增广，注入 Lagrangian 惩罚项。
    用于包装 SB3 env（通过 VecEnv 的 step 钩子实现）。
    """

    def __init__(self, lagrangian: "LagrangianPPO"):
        self.lagrangian = lagrangian

    def augment(self, reward: float, info: dict) -> float:
        cost = self.lagrangian.compute_cost(info)
        return self.lagrangian.compute_augmented_reward(reward, cost)


class LagrangianPPO:
    """
    Lagrangian PPO 核心类。
    封装 SB3 PPO + Lagrangian 对偶变量 + Cost Value Network。
    """

    def __init__(
        self,
        env,
        obs_dim: int,
        cost_limit: float = 1e-4,
        lagrangian_lr: float = 1e-7,
        initial_lambda: float = 10.0,
        hidden: int = 128,
        actor_lr: float = 8e-5,
        critic_lr: float = 1e-3,
        gamma: float = 0.4,
        n_steps: int = 500,
        batch_size: int = 64,
        device: str = "cpu",
        verbose: int = 1,
    ):
        self.cost_limit = cost_limit
        self.lagrangian_lr = lagrangian_lr
        self.lagrangian_multiplier = initial_lambda

        self.cost_critic = CostNetwork(obs_dim, hidden=hidden).to(device)
        self.cost_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=critic_lr)
        self.device = device

        self.reward_wrapper = LagrangianRewardWrapper(self)

        # SB3 PPO — 使用 MultiInputPolicy 支持字典观测
        self.ppo = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=actor_lr,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            verbose=verbose,
            device=device,
        )

        self._lagrangian_cb = LagrangianCallback(self, verbose=verbose)

    def compute_augmented_reward(self, reward: float, cost: float) -> float:
        """Lagrangian augmented reward: r - λ * c"""
        return reward - self.lagrangian_multiplier * cost

    def update_lagrangian(self, mean_cost: float) -> None:
        """对偶变量梯度上升更新（带 ReLU 投影到非负域）"""
        delta = mean_cost - self.cost_limit
        self.lagrangian_multiplier = max(0.0, self.lagrangian_multiplier + self.lagrangian_lr * delta)

    def compute_cost(self, info: dict) -> float:
        """
        从环境 info 中提取 SLA violation indicator。
        cost = 0.4 * hp_viol_count + 0.6 * total_viol_count
        """
        hp_viol = 0
        total_viol = 0
        for k, v in info.items():
            if k.startswith("violation/slice_") and float(v) > 0:
                total_viol += 1
                # 判断是否为 HP slice（通过 meta 键）
                slice_num = k.replace("violation/slice_", "").split("_")[0]
                prio_key = f"meta/slice_{slice_num}_priority"
                if info.get(prio_key, 0) > 0:
                    hp_viol += 1
        return 0.4 * hp_viol + 0.6 * total_viol

    def learn(self, total_timesteps: int) -> None:
        """训练主入口：SB3 PPO 训练 + Lagrangian 回调"""
        self.ppo.learn(
            total_timesteps=total_timesteps,
            callback=self._lagrangian_cb,
        )

    def predict(self, obs, deterministic: bool = True):
        return self.ppo.predict(obs, deterministic=deterministic)

    def save(self, path: str) -> None:
        self.ppo.save(path + "_ppo")
        torch.save({
            "lagrangian_multiplier": self.lagrangian_multiplier,
            "cost_critic_state_dict": self.cost_critic.state_dict(),
        }, path + "_lagrangian.pt")

    @classmethod
    def load(cls, path: str, env, obs_dim: int, device: str = "cpu", **kwargs):
        instance = cls(env=env, obs_dim=obs_dim, device=device, **kwargs)
        instance.ppo = PPO.load(path + "_ppo", env=env, device=device)
        ckpt = torch.load(path + "_lagrangian.pt", map_location=device)
        instance.lagrangian_multiplier = ckpt["lagrangian_multiplier"]
        instance.cost_critic.load_state_dict(ckpt["cost_critic_state_dict"])
        return instance
