from typing import Callable, Optional, Tuple, Type, TypeVar, Union, Any

from hydra.utils import get_class

import numpy as np
from gymnasium import spaces
from omegaconf import OmegaConf
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from src.basic_apis.ppo.ppo_baseline.env_sb3 import CommunicationEnv


class MARLCommEnv(MultiAgentEnv):
    def __init__(
            self,
            cfg: dict
            # *args,
            # **kwargs,
    ):
        # print(kwargs)
        self.comm_env = CommunicationEnv(cfg)
        self.agents = {
            "player_" + str(r)
            for r in range(self.comm_env.components.slices.max_number_slices + 1)
        }
        self._agent_ids = set(self.agents)
        self._obs_space_in_preferred_format = True
        self.observation_space = self.comm_env.observation_space
        self._action_space_in_preferred_format = True
        self.action_space = self.comm_env.action_space

        super().__init__()

    def reset(
            self,
            seed: Optional[int] = None,
            options: dict = {"initial_episode": -1},
            return_info: bool = True,
    ):
        if options is None or options == {}:
            options = {"initial_episode": -1}
        obs, _ = self.comm_env.reset(seed=seed, options=options)
        return obs, {}

    def step(self, action_dict):
        obs, rewards, terminated, truncated, info = {}, {}, {}, {}, {}
        (
            obs,
            rewards,
            termination,
            truncation,
            info,
        ) = self.comm_env.step(action_dict)
        if termination:
            if isinstance(action_dict, dict):
                terminated = {agent: True for agent in self.agents}
                truncated = {agent: True for agent in self.agents}
                terminated["__all__"], truncated["__all__"] = True, True
            else:
                terminated, truncated = True, True
        else:
            if isinstance(action_dict, dict):
                terminated = {agent: False for agent in self.agents}
                truncated = {agent: False for agent in self.agents}
                terminated["__all__"], truncated["__all__"] = False, False
            else:
                terminated = False
                truncated = False

        # RLlib MultiAgentEnv 要求 info 的 key 必须是 agent ids 的子集
        # （或使用 "__common__" 存放全局信息）。
        # 当前业务环境返回的是扁平全局字典（meta/drift/...），这里统一包装。
        if isinstance(action_dict, dict) and isinstance(info, dict):
            if not set(info.keys()).issubset(set(obs.keys())):
                info = {"__common__": info}

        return obs, rewards, terminated, truncated, info

    def set_agent_functions(
            self,
            obs_space_format: Optional[
                Callable[[dict], Union[np.ndarray, dict]]
            ] = None,
            action_format: Optional[
                Callable[[Union[np.ndarray, dict]], np.ndarray]
            ] = None,
            calculate_reward: Optional[
                Callable[[dict], Union[float, dict]]
            ] = None,
            obs_space: spaces.Space = spaces.Space(),
            action_space: spaces.Space = spaces.Space(),
    ):
        self.observation_space = obs_space
        self.action_space = action_space
        self.comm_env.set_agent_functions(
            obs_space_format,
            action_format,
            calculate_reward,
            obs_space,
            action_space,
        )


def env_creator(env_config, only_env=True) -> Union['MARLCommEnv', Tuple['MARLCommEnv', Any]]:
    """
    适配重构后环境的Hydra环境创建器

    Args:
        env_config: Hydra配置字典
        only_env: 是否只返回环境（不返回agent）

    Returns:
        环境实例或环境和agent的元组
    """

    mode = env_config["mode"]
    scenario_mode = env_config["scenario_mode"]
    env_config['state_config']['scenario_skip_episodes'] = env_config[scenario_mode]['scenario_skip_episodes']
    env_config['state_config']['initial_episode'] = env_config[scenario_mode][mode]['initial_episode']
    env_config['state_config']['max_episode'] = env_config[scenario_mode][mode]['max_episode']
    env_config['state_config']['active_scenario_list'] = env_config[scenario_mode][mode]['active_scenario_list']

    # env_config["channel_class"] = get_class(env_config["channel_class"][scenario_mode])
    # env_config["traffic_class"] = get_class(env_config["traffic_class"])
    # env_config["mobility_class"] = get_class(env_config["mobility_class"])
    # env_config["association_class"] = get_class(env_config["association_class"][scenario_mode])

    if not isinstance(env_config["channel_class"], type):
        # 还是字符串或字典，需要转换
        env_config["channel_class"] = get_class(env_config["channel_class"][scenario_mode])

    if not isinstance(env_config["traffic_class"], type):
        env_config["traffic_class"] = get_class(env_config["traffic_class"])

    if not isinstance(env_config["mobility_class"], type):
        env_config["mobility_class"] = get_class(env_config["mobility_class"])

    if not isinstance(env_config["association_class"], type):
        env_config["association_class"] = get_class(env_config["association_class"][scenario_mode])

    marl_comm_env = MARLCommEnv(env_config)

    eval_env = None
    if env_config.get("enable_evaluation", False):
        mode = "evaluating"
        scenario_mode = env_config["scenario_mode"]
        env_config['state_config']['initial_episode'] = env_config[scenario_mode][mode]['initial_episode']
        env_config['state_config']['max_episode'] = env_config[scenario_mode][mode]['max_episode']
        env_config['state_config']['active_scenario_list'] = env_config[scenario_mode][mode]['active_scenario_list']

        eval_env = MARLCommEnv(env_config)

    # ===============================================
    # 5. 创建智能体（如果需要）
    # ===============================================
    agent = None
    if env_config.get("rl", True):
        # 获取智能体类
        agent_class = get_class(env_config["agent_class"])

        if env_config.get("rl"):
            # RL模式 - 使用重构后的便利属性访问方法
            agent = agent_class(
                env=marl_comm_env,
                max_number_ues=marl_comm_env.comm_env.components.slices.max_number_ues,  # 使用便利属性
                max_number_slices=marl_comm_env.comm_env.components.slices.max_number_slices,
                max_number_basestations=marl_comm_env.comm_env.config.basestation_config.max_number_basestations,
                num_available_rbs=marl_comm_env.comm_env.config.basestation_config.num_available_rbs,
                eval_env=eval_env if env_config.get("enable_evaluation", False) else None,
                agent_name=env_config.get("agent_name", "default_agent"),
                seed=env_config.get("seed", 42),
                episode_evaluation_freq=env_config.get("episode_evaluation_freq", 10),
                number_evaluation_episodes=env_config.get("number_evaluation_episodes", 5),
                checkpoint_episode_freq=env_config.get("checkpoint_episode_freq", 10),
                eval_initial_env_episode=env_config.get("eval_initial_env_episode", 0),
            )
        else:
            # 非RL模式
            agent = agent_class(
                marl_comm_env,
                marl_comm_env.comm_env.components.slices.max_number_ues,
                marl_comm_env.comm_env.components.slices.max_number_slices,
                marl_comm_env.comm_env.config.basestation_config.max_number_basestations,
                marl_comm_env.comm_env.config.basestation_config.num_available_rbs,
                seed=env_config.get("seed", 42),
            )

        # ===============================================
        # 6. 设置智能体函数
        # ===============================================
        marl_comm_env.set_agent_functions(
            obs_space_format=agent.obs_space_format,
            action_format=agent.action_format,
            calculate_reward=agent.calculate_reward,
            obs_space=agent.get_obs_space(),
            action_space=agent.get_action_space(),
        )


        if eval_env is not None:
            eval_env.set_agent_functions(
                obs_space_format=agent.obs_space_format,
                action_format=agent.action_format,
                calculate_reward=agent.calculate_reward,
                obs_space=agent.get_obs_space(),
                action_space=agent.get_action_space(),
            )

        # 初始化智能体
        if hasattr(agent, 'init_agent'):
            agent.init_agent()

    # ===============================================
    # 7. 返回结果
    # ===============================================
    if only_env:
        return marl_comm_env
    else:
        return marl_comm_env, agent
