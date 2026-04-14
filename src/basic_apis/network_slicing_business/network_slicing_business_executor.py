import numpy as np
from typing import Tuple, Type
from dataclasses import dataclass
from copy import deepcopy

from .components import UEs, Association, Basestations, Channel, Metrics, Mobility, Slices, Traffic, Buffer

from .path_manager import PathManager


@dataclass
class BasestationConfig:
    """基站配置"""
    max_number_basestations: int
    bandwidths: np.ndarray
    carrier_frequencies: np.ndarray
    num_available_rbs: np.ndarray
    basestation_ue_assoc: np.ndarray
    basestation_slice_assoc: np.ndarray
    noise_power: float

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
            basestation_slice_assoc=bs_slice_assoc,
            noise_power=10e-14
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


class ComponentConfig:
    """统一的环境配置管理器"""

    def __init__(self, components_settsings: dict):
        self.components_settsings = components_settsings
        self.ue_config = None
        self.slice_config = None
        self.basestation_config = None

        self._parse_config()

    def _parse_config(self):
        """解析和验证配置"""
        # 先解析UE配置，因为其他配置可能依赖它
        self.ue_config = UEConfig.from_dict(self.components_settsings["ues"])

        # 解析切片配置
        self.slice_config = SliceConfig.from_dict(
            self.components_settsings["slices"],
            self.ue_config.max_number_ues
        )

        # 解析基站配置
        self.basestation_config = BasestationConfig.from_dict(
            self.components_settsings["basestations"],
            self.ue_config.max_number_ues,
            self.slice_config.max_number_slices
        )

    def validate(self) -> bool:
        """验证配置的一致性"""
        # 添加配置验证逻辑
        assert self.ue_config.max_number_ues > 0
        assert self.basestation_config.max_number_basestations > 0
        assert self.slice_config.max_number_slices > 0
        # assert self.simulation_config.max_number_steps > 0
        return True

@dataclass
class Components:
    """多智能体场景组件容器"""
    ues: UEs
    associations: Association
    slices: Slices
    basestations: Basestations
    mobility: Mobility
    channel: Channel
    traffic: Traffic
    metrics: Metrics

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)


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
    """多智能体组件工厂"""

    def __init__(
         self,
         component_config: ComponentConfig,
         component_classes: ComponentClasses,
         np_random,
         path_manager: PathManager = None
        ):
        self.component_config = component_config
        self.component_classes = component_classes
        self.np_random = np_random
        self.path_manager = path_manager

    def create_scenario_components(self,
                                   episode_number: int,
                                   step_number: int) -> Components:
        """创建多智能体场景所需的所有组件"""

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


        return Components(
            ues=ues,
            associations=associations,
            slices=slices,
            basestations=basestations,
            mobility=mobility,
            channel=channel,
            traffic=traffic,
            metrics=metrics,
        )

    # 其他方法保持不变
    def _create_ues(self) -> UEs:
        """创建UE组件"""
        return UEs(
            self.component_config.ue_config.max_number_ues,
            self.component_config.ue_config.max_buffer_latencies,
            self.component_config.ue_config.max_buffer_pkts,
            self.component_config.ue_config.pkt_sizes,
        )

    def _create_associations(self, ues: UEs, episode_number: int,
                             step_number: int) -> Association:
        """创建关联组件"""
        return self.component_classes.AssociationClass(
            ues,
            self.component_config.ue_config.max_number_ues,
            self.component_config.basestation_config.max_number_basestations,
            self.component_config.slice_config.max_number_slices,
            self.np_random,
            path_manager=self.path_manager,
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
            self.component_config.basestation_config.basestation_ue_assoc.copy(),
            self.component_config.basestation_config.basestation_slice_assoc.copy(),
            self.component_config.slice_config.slice_ue_assoc.copy(),
            self.component_config.slice_config.slice_req.copy(),
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
            self.component_config.slice_config.max_number_slices,
            self.component_config.ue_config.max_number_ues,
            assoc_data['slice_ue_assoc'],
            assoc_data['slice_req'],
        )

    def _create_basestations(self, assoc_data: dict) -> Basestations:
        """创建基站组件"""
        return Basestations(
            assoc_data['bs_slice_assoc'],
            assoc_data['bs_ue_assoc'],
            self.component_config.basestation_config.bandwidths,
        )

    def _create_mobility(self) -> Mobility:
        """创建移动性组件"""
        return self.component_classes.MobilityClass(
            self.component_config.ue_config.max_number_ues,
        )

    def _create_channel(self) -> Channel:
        """创建信道组件"""
        return self.component_classes.ChannelClass(
            self.component_config.basestation_config.num_available_rbs,
            path_manager=self.path_manager,
        )

    def _create_traffic(self) -> Traffic:
        """创建流量组件"""
        return self.component_classes.TrafficClass(
            self.component_config.ue_config.max_number_ues,
            rng=self.np_random,
        )

    def _create_metrics(self) -> Metrics:
        """创建指标组件"""
        return Metrics(self.path_manager)



class NetworkSlicingBusinessExecutor:
    """
    [One-Shot Adapter] 负责执行多智能体环境的物理层计算逻辑

    变化点：
    1. 不再处理 Action 解析 (移交给了 Env)。
    2. 不再处理 Power Scaling (移交给了 Env)。
    3. 移除了 Accumulative Step 逻辑，改为单次 TTI 全局执行。
    """

    def __init__(self, components: Components, component_config: ComponentConfig):
        self.components = components
        self.component_config = component_config

        # === 1. 资源分配矩阵 ===
        # Shape: [Num_Users, Num_Phys_RBs] (例如 25 x 135)
        # 这个矩阵由 Environment 在 step() 中直接填充
        self.num_phys_rbs = int(self.component_config.basestation_config.num_available_rbs[0])
        self.temp_rb_allocation = np.zeros(
            (self.components.ues.max_number_ues, self.num_phys_rbs),
            dtype=np.float32
        )
        self.temp_rb_ues_association = np.zeros(
            (1, self.components.ues.max_number_ues, self.num_phys_rbs),
            dtype=np.int8
        )

    def execute_tti_physics(self, channel_timestep: int, episode_number: int) -> dict:
        """
        执行一个完整的 TTI (1ms) 物理层模拟。
        """

        # 1. Mobility Step
        mobilities = self.components.mobility.step(channel_timestep, episode_number)

        # 2. Channel Step (使用 Env 填好的功率分配矩阵计算 SINR)
        spectral_efficiencies, target_cell_power = self.components.channel.step(
            channel_timestep, episode_number, self.temp_rb_allocation
        )

        # 3. Traffic Step
        traffics = self.components.traffic.step(
            self.components.slices.ue_assoc,
            self.components.slices.requirements,
            channel_timestep,
            episode_number,
        )

        # 4. UEs Step (使用处理后的流量更新 Buffer)
        step_hist = self.components.ues.step(
            self.temp_rb_ues_association,
            traffics,
            spectral_efficiencies,
            self.components.basestations.bandwidths,
            self.component_config.basestation_config.num_available_rbs,
            update_state=True
        )

        # 5. 数据聚合
        # 这样 Environment 就可以在 info 里读取它了
        return {
            **step_hist,
            "mobility": mobilities,
            "spectral_efficiencies": spectral_efficiencies,
            "sched_decision": self.temp_rb_allocation,
            "target_cell_power": target_cell_power,
            "basestation_ue_assoc": self.components.basestations.ue_assoc,
            "basestation_slice_assoc": self.components.basestations.slice_assoc,
            "slice_ue_assoc": self.components.slices.ue_assoc,
            "slice_req": self.components.slices.requirements,

            # [新增] 将实际注入 Buffer 的流量数据暴露出来
            "pkt_incoming_bits": traffics
        }

    def update_associations(self, step_number: int, episode_number: int):
        """更新关联关系 (保持不变)"""
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

    def step_metric(self, hist_data):
        self.components.metrics.step(hist_data)

    def save_metric(self, *path_parts):
        self.components.metrics.save(*path_parts)

    def reset_metric(self):
        self.components.metrics.reset()
