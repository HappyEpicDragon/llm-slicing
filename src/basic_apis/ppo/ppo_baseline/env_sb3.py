from itertools import cycle
from typing import Callable, Optional, Tuple, Type, Union, Dict
from dataclasses import dataclass

import gymnasium as gym
import numpy as np


from src.basic_apis.network_slicing_business.components import Association, Basestations, Channel, Metrics, Mobility, Slices, Traffic, UEs
from src.basic_apis.network_slicing_business.path_context import PathContext


# =====================================================================
# 1. 配置管理器 - 统一处理所有配置
# =====================================================================

@dataclass
class BasestationConfig:
    """基站配置"""
    max_number_basestations: int
    bandwidths: np.ndarray
    carrier_frequencies: np.ndarray
    num_available_rbs: np.ndarray
    basestation_ue_assoc: np.ndarray
    basestation_slice_assoc: np.ndarray

    @classmethod
    def from_dict(cls, config_dict: dict, max_ues: int, max_slices: int):
        max_bs = config_dict["max_number_basestations"]

        # 处理关联矩阵的默认值
        bs_ue_assoc = (
            np.array(config_dict["basestation_ue_assoc"])
            if config_dict.get("basestation_ue_assoc") is not None
            else np.ones((max_bs, max_ues))
        )

        bs_slice_assoc = (
            np.array(config_dict["basestation_slice_assoc"])
            if config_dict.get("basestation_slice_assoc") is not None
            else np.ones((max_bs, max_slices))
        )

        return cls(
            max_number_basestations=max_bs,
            bandwidths=np.array(config_dict["bandwidths"]),
            carrier_frequencies=np.array(config_dict["carrier_frequencies"]),
            num_available_rbs=np.array(config_dict["num_available_rbs"]),
            basestation_ue_assoc=bs_ue_assoc,
            basestation_slice_assoc=bs_slice_assoc
        )


@dataclass
class UEConfig:
    """UE配置"""
    max_number_ues: int
    max_buffer_latencies: np.ndarray
    max_buffer_pkts: np.ndarray
    pkt_sizes: np.ndarray

    @classmethod
    def from_dict(cls, config_dict: dict):
        max_ues = config_dict["max_number_ues"]

        return cls(
            max_number_ues=max_ues,
            max_buffer_latencies=(
                np.array(config_dict["max_buffer_latencies"])
                if config_dict.get("max_buffer_latencies") is not None
                else np.ones(max_ues, dtype=int) * 100
            ),
            max_buffer_pkts=(
                np.array(config_dict["max_buffer_pkts"])
                if config_dict.get("max_buffer_pkts") is not None
                else np.ones(max_ues) * 1024 * 10
            ),
            pkt_sizes=(
                np.array(config_dict["pkt_sizes"])
                if config_dict.get("pkt_sizes") is not None
                else np.ones(max_ues) * 8192 * 8
            )
        )


@dataclass
class SliceConfig:
    """切片配置"""
    max_number_slices: int
    slice_ue_assoc: np.ndarray
    slice_req: dict

    @classmethod
    def from_dict(cls, config_dict: dict, max_ues: int):
        max_slices = config_dict["max_number_slices"]

        return cls(
            max_number_slices=max_slices,
            slice_ue_assoc=(
                np.array(config_dict["slice_ue_assoc"])
                if config_dict.get("slice_ue_assoc") is not None
                else np.ones((max_slices, max_ues))
            ),
            slice_req=(
                config_dict["slice_req"]
                if config_dict.get("slice_req") is not None
                else {}
            )
        )


class EnvironmentConfig:
    """统一的环境配置管理器"""

    def __init__(self, env_settings: dict, **overrides):
        self.env_settings = env_settings
        self.overrides = overrides
        self._parse_config()

    def _parse_config(self):
        """解析和验证配置"""
        # 先解析UE配置，因为其他配置可能依赖它
        self.ue_config = UEConfig.from_dict(self.env_settings["ues"])

        # 解析切片配置
        self.slice_config = SliceConfig.from_dict(
            self.env_settings["slices"],
            self.ue_config.max_number_ues
        )

        # 解析基站配置
        self.basestation_config = BasestationConfig.from_dict(
            self.env_settings["basestations"],
            self.ue_config.max_number_ues,
            self.slice_config.max_number_slices
        )


    def validate(self) -> bool:
        """验证配置的一致性"""
        # 添加配置验证逻辑
        assert self.ue_config.max_number_ues > 0
        assert self.basestation_config.max_number_basestations > 0
        assert self.slice_config.max_number_slices > 0
        return True


# =====================================================================
# 2. 组件容器和工厂
# =====================================================================

@dataclass
class ScenarioComponents:
    """场景组件容器"""
    ues: UEs
    associations: Association
    slices: Slices
    basestations: Basestations
    mobility: Mobility
    channel: Channel
    traffic: Traffic
    metrics: Metrics

    def get_association_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取关联矩阵"""
        return (
            self.basestations.ue_assoc,
            self.basestations.slice_assoc,
            self.slices.ue_assoc
        )

    def update_association_matrices(self, bs_ue_assoc: np.ndarray,
                                    bs_slice_assoc: np.ndarray,
                                    slice_ue_assoc: np.ndarray,
                                    slice_req: dict):
        """更新关联矩阵"""
        self.basestations.ue_assoc = bs_ue_assoc
        self.basestations.slice_assoc = bs_slice_assoc
        self.slices.ue_assoc = slice_ue_assoc
        self.slices.requirements = slice_req


@dataclass
class ComponentClasses:
    """组件类集合"""
    ChannelClass: Type[Channel]
    TrafficClass: Type[Traffic]
    MobilityClass: Type[Mobility]
    AssociationClass: Type[Association]


class ComponentFactory:
    """组件工厂，管理复杂的组件创建逻辑"""

    def __init__(self,
                 config: EnvironmentConfig,
                 component_classes: ComponentClasses,
                 np_random,
                 path_context: PathContext = None,):
        self.config = config
        self.component_classes = component_classes
        self.np_random = np_random
        self.path_context = path_context

    def create_scenario_components(self,
                                   episode_number: int,
                                   step_number: int) -> ScenarioComponents:
        """创建场景所需的所有组件"""

        # 1. 创建基础组件
        ues = self._create_ues()
        associations = self._create_associations(ues, episode_number, step_number)

        # 2. 获取初始关联关系
        assoc_data = self._get_initial_associations(
            associations, episode_number, step_number
        )

        # 3. 创建依赖组件
        slices = self._create_slices(assoc_data)
        basestations = self._create_basestations(assoc_data)

        # 4. 创建其他组件
        mobility = self._create_mobility()
        channel = self._create_channel()
        traffic = self._create_traffic()
        metrics = self._create_metrics()

        return ScenarioComponents(
            ues=ues,
            associations=associations,
            slices=slices,
            basestations=basestations,
            mobility=mobility,
            channel=channel,
            traffic=traffic,
            metrics=metrics
        )

    def _create_ues(self) -> UEs:
        """创建UE组件"""
        return UEs(
            self.config.ue_config.max_number_ues,
            self.config.ue_config.max_buffer_latencies,
            self.config.ue_config.max_buffer_pkts,
            self.config.ue_config.pkt_sizes,
        )

    def _create_associations(self, ues: UEs, episode_number: int,
                             step_number: int) -> Association:
        """创建关联组件"""
        return self.component_classes.AssociationClass(
            ues,
            self.config.ue_config.max_number_ues,
            self.config.basestation_config.max_number_basestations,
            self.config.slice_config.max_number_slices,
            self.np_random,
            path_context=self.path_context,
        )

    def _get_initial_associations(self, associations: Association,
                                  episode_number: int, step_number: int) -> dict:
        """获取初始关联关系"""
        (
            bs_ue_assoc,
            bs_slice_assoc,
            slice_ue_assoc,
            slice_req,
        ) = associations.step(
            self.config.basestation_config.basestation_ue_assoc.copy(),
            self.config.basestation_config.basestation_slice_assoc.copy(),
            self.config.slice_config.slice_ue_assoc.copy(),
            self.config.slice_config.slice_req.copy(),
            step_number,
            episode_number,
        )

        return {
            'bs_ue_assoc': bs_ue_assoc,
            'bs_slice_assoc': bs_slice_assoc,
            'slice_ue_assoc': slice_ue_assoc,
            'slice_req': slice_req
        }

    def _create_slices(self, assoc_data: dict) -> Slices:
        """创建切片组件"""
        return Slices(
            self.config.slice_config.max_number_slices,
            self.config.ue_config.max_number_ues,
            assoc_data['slice_ue_assoc'],
            assoc_data['slice_req'],
        )

    def _create_basestations(self, assoc_data: dict) -> Basestations:
        """创建基站组件"""
        return Basestations(
            assoc_data['bs_slice_assoc'],
            assoc_data['bs_ue_assoc'],
            self.config.basestation_config.bandwidths,
        )

    def _create_mobility(self) -> Mobility:
        """创建移动性组件"""
        return self.component_classes.MobilityClass(
            self.config.ue_config.max_number_ues,
        )

    def _create_channel(self) -> Channel:
        """创建信道组件"""
        return self.component_classes.ChannelClass(
            self.config.basestation_config.num_available_rbs,
            path_context=self.path_context,
        )

    def _create_traffic(self) -> Traffic:
        """创建流量组件"""
        return self.component_classes.TrafficClass(
            self.config.ue_config.max_number_ues,
            rng=self.np_random,
        )

    def _create_metrics(self) -> Metrics:
        """创建指标组件"""
        return Metrics(self.path_context)


# =====================================================================
# 3. 环境状态管理器
# =====================================================================

class EnvironmentState:
    """管理环境状态和episode逻辑"""

    def __init__(self, config: dict):
        self.config = config
        self.step_number = 0
        self.episode_interval_start = config['initial_episode']
        self.episode_interval_end = config['max_episode']
        self.active_scenario_list = config['active_scenario_list']
        self.scenario_skip_episodes = config['scenario_skip_episodes']
        self.max_number_steps = config['max_number_steps']
        self.current_scenario_idx = 0
        self.initial_episode = self.active_scenario_list[self.current_scenario_idx] * self.scenario_skip_episodes + self.episode_interval_start
        self.max_episode = self.active_scenario_list[self.current_scenario_idx] * self.scenario_skip_episodes + self.episode_interval_end
        self.episode_number = self.initial_episode


    def reset(self, initial_episode: int = -1):
        """重置环境状态"""
        if initial_episode != -1:
            self.config['initial_episode'] = initial_episode
            self.episode_number = initial_episode

        elif self.step_number == self.max_number_steps:
            self._advance_episode()

        self.step_number = 0

    def step(self) -> bool:
        """前进一步，返回是否episode结束"""
        self.step_number += 1
        return self.is_episode_done()

    def is_episode_done(self) -> bool:
        """检查episode是否结束"""
        return self.step_number >= self.max_number_steps

    def _advance_episode(self):
        """推进到下一个episode"""
        if self.episode_number < (self.max_episode - 1):
            self.episode_number += 1
        elif self.episode_number == (self.max_episode - 1):
            if self.current_scenario_idx == len(self.active_scenario_list) - 1:
                self.current_scenario_idx = 0
            else:
                self.current_scenario_idx += 1
            self.initial_episode = self.active_scenario_list[self.current_scenario_idx] * self.scenario_skip_episodes + self.episode_interval_start
            self.max_episode = self.active_scenario_list[self.current_scenario_idx] * self.scenario_skip_episodes + self.episode_interval_end
            self.episode_number = self.initial_episode
        else:
            raise Exception(
                f"Episode number received a non expected value equals to {self.episode_number}. "
                f"Max episode number: {self.max_episode}"
            )


# =====================================================================
# 4. Agent函数管理器
# =====================================================================

class AgentFunctions:
    """管理agent相关的函数"""

    def __init__(self, functions: Dict[str, Callable] = None):
        if functions is None:
            functions = {}

        self.format_observation = functions.get(
            'obs_space_format', self._default_obs_format
        )
        self.calculate_reward = functions.get(
            'calculate_reward', self._default_reward
        )
        self.format_action = functions.get(
            'action_format', self._default_action_format
        )

    @staticmethod
    def _default_obs_format(obs_space: dict) -> Union[np.ndarray, dict]:
        """默认观察空间格式化"""
        print("Calling default obs_space_format")
        return np.array(list(obs_space.items()), dtype=object)

    @staticmethod
    def _default_reward(obs_space: dict) -> float:
        """默认奖励计算"""
        return 0.0

    @staticmethod
    def _default_action_format(action: Union[np.ndarray, dict]) -> np.ndarray:
        """默认动作格式化"""
        return np.array(action)

    def update_functions(self, **functions):
        """更新agent函数"""
        if 'obs_space_format' in functions:
            self.format_observation = functions['obs_space_format']
        if 'calculate_reward' in functions:
            self.calculate_reward = functions['calculate_reward']
        if 'action_format' in functions:
            self.format_action = functions['action_format']


# =====================================================================
# 5. 步骤执行器
# =====================================================================

class StepExecutor:
    """负责执行环境步骤的逻辑"""

    def __init__(self, components: ScenarioComponents, config: EnvironmentConfig):
        self.components = components
        self.config = config

    def execute_step(self, formatted_action: np.ndarray,
                     step_number: int, episode_number: int) -> dict:
        """执行单步环境逻辑"""

        # 1. 获取移动性数据
        mobilities = self.components.mobility.step(step_number, episode_number)

        # 2. 计算频谱效率
        spectral_efficiencies = self.components.channel.step(
            step_number, episode_number
        )

        # 3. 生成流量
        traffics = self.components.traffic.step(
            self.components.slices.ue_assoc,
            self.components.slices.requirements,
            step_number,
            episode_number,
        )

        # 4. UE缓冲区处理
        step_hist = self.components.ues.step(
            formatted_action,
            traffics,
            spectral_efficiencies,
            self.components.basestations.bandwidths,
            self.config.basestation_config.num_available_rbs,
        )

        # 5. 聚合所有步骤数据
        return self._aggregate_step_data(
            step_hist, mobilities, spectral_efficiencies, formatted_action
        )

    def _aggregate_step_data(self, step_hist: dict, mobilities: np.ndarray,
                             spectral_efficiencies: np.ndarray,
                             action: np.ndarray) -> dict:
        """聚合步骤数据"""
        return {
            **step_hist,
            "mobility": mobilities,
            "spectral_efficiencies": spectral_efficiencies,
            "basestation_ue_assoc": self.components.basestations.ue_assoc,
            "basestation_slice_assoc": self.components.basestations.slice_assoc,
            "slice_ue_assoc": self.components.slices.ue_assoc,
            "sched_decision": action,
            "slice_req": self.components.slices.requirements,
        }

    def update_associations(self, step_number: int, episode_number: int):
        """更新关联关系"""
        (
            bs_ue_assoc,
            bs_slice_assoc,
            slice_ue_assoc,
            slice_requirements,
        ) = self.components.associations.step(
            self.components.basestations.ue_assoc,
            self.components.basestations.slice_assoc,
            self.components.slices.ue_assoc,
            self.components.slices.requirements,
            step_number,
            episode_number,
        )

        self.components.update_association_matrices(
            bs_ue_assoc, bs_slice_assoc, slice_ue_assoc, slice_requirements
        )


# =====================================================================
# 6. 主环境类 (重构后)
# =====================================================================

class CommunicationEnv(gym.Env):
    """重构后的通信环境类 - 专注于gym接口实现"""

    metadata = {"render.modes": ["human"]}

    def __init__(self, cfg) -> None:

        # 设置随机种子
        self.cfg = cfg
        self.seed = cfg['seed'] if not self.cfg['mode'] == 'testing' else cfg['seed_test']
        mode = cfg['mode']
        scenario_mode = cfg['scenario_mode']
        self.save_hist = cfg[scenario_mode][mode]['save_hist']
        self.model_name = cfg['model_name']


        config_overrides = {}
        self.config = EnvironmentConfig(cfg['env_settings'], **config_overrides)
        self.config.validate()

        self.state_config = cfg['state_config']
        self.state_config['scenario_skip_episodes'] = cfg[scenario_mode]['scenario_skip_episodes']
        self.state_config['initial_episode'] = cfg[scenario_mode][mode]['initial_episode']
        self.state_config['max_episode'] = cfg[scenario_mode][mode]['max_episode']
        self.state_config['active_scenario_list'] = cfg[scenario_mode][mode]['active_scenario_list']

        # 初始化环境状态
        self.env_state = EnvironmentState(self.state_config)

        # 设置组件类
        self.component_classes = ComponentClasses(
            ChannelClass=cfg['channel_class'],
            TrafficClass=cfg['traffic_class'],
            MobilityClass=cfg['mobility_class'],
            AssociationClass=cfg['association_class']
        )

        self.path_context = cfg['path_context']

        agent_functions = {}
        self.agent_functions = AgentFunctions(agent_functions)

        # 设置gym空间
        self.observation_space = gym.spaces.Space()
        self.action_space = gym.spaces.Space()

        # 初始化组件
        self.components = None
        self.step_executor = None

        self.component_factory = ComponentFactory(
            self.config, self.component_classes, self.np_random, self.path_context
        )

        self._create_scenario()
        self.metrics_save_path_parts = ()
        self.ckp_idx = cfg.get('ckp_idx', None)

    def update_metric_save_path_parts(self, new_metric_save_path_parts: tuple):
        self.metrics_save_path_parts = new_metric_save_path_parts

    def step(
            self, action: Union[np.ndarray, dict]
    ) -> Tuple[Union[np.ndarray, dict], Union[float, dict], bool, bool, dict]:
        """执行一步环境交互"""

        # 1. 格式化动作
        formatted_action = self.agent_functions.format_action(action)

        # 3. 执行环境步骤
        step_data = self.step_executor.execute_step(
            formatted_action,
            self.env_state.step_number,
            self.env_state.episode_number
        )

        # 4. 计算观察和奖励
        observation = self.agent_functions.format_observation(step_data)
        reward = self.agent_functions.calculate_reward(step_data)

        # 5. 更新环境状态
        is_done = self.env_state.step()

        # 6. 保存历史数据和更新关联
        step_data.update({"reward": reward, "obs": observation, "agent_action": action})
        self.components.metrics.step(step_data)

        if is_done:
            if self.save_hist:
                if self.ckp_idx is not None:
                    self.update_metric_save_path_parts([f"checkpoint", self.model_name, f"ep_{self.env_state.episode_number}.npz"])
                else:
                    self.update_metric_save_path_parts([self.model_name, f"ep_{self.env_state.episode_number}.npz"])
                self._save_history()
        else:
            # 更新关联关系
            self.step_executor.update_associations(
                self.env_state.step_number,
                self.env_state.episode_number
            )

        return observation, reward, is_done, False, self._build_drift_info()

    def _build_drift_info(self) -> dict:
        """从 IBSched agent 缓存的 intent_drift 构建与 HierarchicalSlicingEnv 格式一致的 info dict。
        若 agent 未缓存 drift（如纯规则调度），返回空字典。
        """
        fmt = self.agent_functions.format_observation
        agent = getattr(fmt, '__self__', None)
        if agent is None:
            return {}
        intent_drift = getattr(agent, '_last_intent_drift', None)
        if intent_drift is None:
            return {}
        slice_req = getattr(agent, '_last_slice_req', {}) or {}
        slice_ue_assoc = getattr(agent, '_last_slice_ue_assoc', None)

        info = {}
        metric_names = ['thr', 'rel', 'lat']
        num_slices = intent_drift.shape[0]

        for s_idx in range(num_slices):
            slice_drift = intent_drift[s_idx]   # (max_ues_per_slice, 3)

            is_active = 0
            if slice_ue_assoc is not None and s_idx < slice_ue_assoc.shape[0]:
                is_active = 1 if np.sum(slice_ue_assoc[s_idx]) > 0 else 0
            info[f"meta/slice_{s_idx}_active"] = is_active

            req = slice_req.get(f'slice_{s_idx}', {})
            prio = req.get('priority', 0)
            info[f"meta/slice_{s_idx}_priority"] = int(prio)

            active_metrics = set()
            for param in req.get('parameters', {}).values():
                name = param.get('name', '').lower()
                if name == 'throughput':
                    active_metrics.add('thr')
                elif name == 'reliability':
                    active_metrics.add('rel')
                elif name == 'latency':
                    active_metrics.add('lat')

            for m_idx, m_name in enumerate(metric_names):
                is_req = 1 if m_name in active_metrics else 0
                info[f"meta/slice_{s_idx}_{m_name}_req"] = is_req
                col = slice_drift[:, m_idx]
                valid_vals = col[col > -1.5]
                if len(valid_vals) > 0 and is_req:
                    info[f"drift/slice_{s_idx}_{m_name}"] = float(np.min(valid_vals))
                else:
                    info[f"drift/slice_{s_idx}_{m_name}"] = 0.0

        return info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[dict] = None
              ) -> Tuple[Union[dict, np.ndarray], dict]:
        """重置环境"""

        if options is None:
            options = {"initial_episode": -1}

        # 处理种子
        if seed is None and self.seed is not None:
            seed = self.seed
            self.seed = None
        elif seed is not None:
            self.seed = None

        # 调用父类reset
        super().reset(seed=seed)

        # 重置状态
        self.env_state.reset(
            options.get("initial_episode", -1)
        )

        self.component_factory = ComponentFactory(
            self.config, self.component_classes, self.np_random, self.path_context
        )

        self._create_scenario()

        # 获取初始观察
        initial_obs = self._get_initial_observation()

        return self.agent_functions.format_observation(initial_obs), {}

    def set_agent_functions(self,
                            obs_space_format: Optional[Callable[[dict], Union[np.ndarray, dict]]] = None,
                            action_format: Optional[Callable[[Union[np.ndarray, dict]], np.ndarray]] = None,
                            calculate_reward: Optional[Callable[[dict], Union[float, dict]]] = None,
                            obs_space: gym.spaces.Space = None,
                            action_space: gym.spaces.Space = None):
        """设置agent相关的函数和空间"""

        if obs_space is not None:
            self.observation_space = obs_space
        if action_space is not None:
            self.action_space = action_space

        update_dict = {}
        if obs_space_format is not None:
            update_dict['obs_space_format'] = obs_space_format
        if action_format is not None:
            update_dict['action_format'] = action_format
        if calculate_reward is not None:
            update_dict['calculate_reward'] = calculate_reward

        if update_dict:
            self.agent_functions.update_functions(**update_dict)

    def _create_scenario(self):
        """创建场景组件"""
        self.components = self.component_factory.create_scenario_components(
            self.env_state.episode_number,
            self.env_state.step_number
        )

        self.components.channel.step = self.components.channel.step_origin

        self.step_executor = StepExecutor(self.components, self.config)

    def _get_initial_observation(self) -> dict:
        """获取初始观察"""
        initial_positions = self.components.mobility.step(
            self.env_state.step_number,
            self.env_state.episode_number
        )

        return {
            "mobility": initial_positions,
            "spectral_efficiencies": self.components.channel.step(
                self.env_state.step_number,
                self.env_state.episode_number,
            ),
            "basestation_ue_assoc": self.components.basestations.ue_assoc,
            "basestation_slice_assoc": self.components.basestations.slice_assoc,
            "slice_ue_assoc": self.components.slices.ue_assoc,
            "sched_decision": np.array([
                np.zeros((
                    self.config.ue_config.max_number_ues,
                    self.config.basestation_config.num_available_rbs[i]
                ))
                for i in range(self.config.basestation_config.max_number_basestations)
            ], dtype=object),
            "pkt_incoming": self.components.traffic.step(
                self.components.slices.ue_assoc,
                self.components.slices.requirements,
                self.env_state.step_number,
                self.env_state.episode_number,
            ),
            "pkt_throughputs": np.zeros(self.config.ue_config.max_number_ues),
            "pkt_effective_thr": np.zeros(self.config.ue_config.max_number_ues),
            "buffer_occupancies": np.zeros(self.config.ue_config.max_number_ues),
            "buffer_latencies": np.zeros(self.config.ue_config.max_number_ues),
            "dropped_pkts": np.zeros(self.config.ue_config.max_number_ues),
            "slice_req": self.components.slices.requirements,
        }

    def _save_history(self):
        """保存历史数据"""
        if self.metrics_save_path_parts == () and not self.ckp_idx:
            self.metrics_save_path_parts = [f"scenario_0", f"ep_{self.env_state.episode_number}"]
        # elif self.metrics_save_path_parts == () and self.ckp_idx:
        #
        self.components.metrics.save(
            *self.metrics_save_path_parts
        )

    def set_agent(self, agent):
        """设置agent实例，用于获取action和observation空间"""
        self.agent = agent
        if hasattr(agent, 'get_action_space'):
            self.action_space = agent.get_action_space()
        if hasattr(agent, 'get_obs_space'):
            self.observation_space = agent.get_obs_space()

    # 兼容性方法
    def save_hist_metrics(self):
        """保存历史指标 (兼容性方法)"""
        self._save_history()

    @staticmethod
    def calculate_reward_default(obs_space: dict) -> float:
        """默认奖励计算 (兼容性方法)"""
        return AgentFunctions._default_reward(obs_space)

    @staticmethod
    def action_format_default(action: Union[np.ndarray, dict]) -> np.ndarray:
        """默认动作格式化 (兼容性方法)"""
        return AgentFunctions._default_action_format(action)

    @staticmethod
    def obs_space_format_default(obs_space: dict) -> Union[np.ndarray, dict]:
        """默认观察格式化 (兼容性方法)"""
        return AgentFunctions._default_obs_format(obs_space)


