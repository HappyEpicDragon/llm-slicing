import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict

import numpy as np
from pathlib import Path
import h5py
from copy import deepcopy

from .path_resolver import association_file_path, channel_mat_path, channel_npz_path, metrics_dir_path


class Buffer:
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(self, max_packets_buffer: int, max_packet_age: int) -> None:
        """
        Parameters
        ----------
        max_packets_buffer: int
            Maximum number of packets allowed in the buffer
        max_packets_age : int
            Maximum latency that packets can wait in the buffer
        """
        self.buffer = np.zeros(max_packet_age + 1, dtype=int)
        self.max_packets_buffer = max_packets_buffer
        self.max_packets_age = max_packet_age
        self.dropped_packets = 0
        self.sent_packets = 0

    def _process_logic(self, buffer_arr, num_packets_arrived, packets_available_to_send):
        """
        纯计算逻辑：输入一个 numpy buffer 数组，返回处理后的状态和统计数据。
        不修改 self 上的任何属性，只操作传入的 buffer_arr。
        """
        # 1. 丢弃超时包 (buffer最后一个元素)
        dropped_packets = buffer_arr[-1]

        # 2. 滚动缓冲区
        # 注意：为了性能，这里不使用 np.roll (会产生新内存拷贝)，可以使用切片操作优化，
        # 但为了保持和你原逻辑一致，这里先用 roll，或者直接手动移位。
        # 为了简单且安全，这里对副本操作，逻辑保持原样：
        buffer_arr = np.roll(buffer_arr, 1)
        buffer_arr[0] = 0

        # 3. 计算占用
        current_occupancy = np.sum(buffer_arr)

        # 4. 接收新包
        available_space = self.max_packets_buffer - current_occupancy
        packets_to_receive = min(num_packets_arrived, available_space)
        dropped_packets += num_packets_arrived - packets_to_receive
        buffer_arr[0] = packets_to_receive

        # 5. 发送包逻辑
        initial_total = np.sum(buffer_arr)
        sent_packets = 0

        if packets_available_to_send > 0 and initial_total > 0:
            cumulative_sum_from_right = np.cumsum(buffer_arr[::-1])[::-1]

            k = len(buffer_arr)
            # 优化搜索：使用 np.searchsorted 或 boolean mask 可能更快，这里保持原逻辑
            # 注意：原逻辑有个循环，这里可以保留
            for i in range(len(cumulative_sum_from_right)):
                if cumulative_sum_from_right[i] >= packets_available_to_send:
                    k = i
                    break

            if k < buffer_arr.shape[0]:
                packets_at_k = packets_available_to_send
                if k + 1 < buffer_arr.shape[0]:
                    packets_at_k -= cumulative_sum_from_right[k + 1]

                buffer_arr[k + 1:] = 0
                buffer_arr[k] -= packets_at_k
            else:
                buffer_arr[:] = 0

            final_total = np.sum(buffer_arr)
            sent_packets = initial_total - final_total

        return buffer_arr, sent_packets, dropped_packets

    def receive_and_send(self, num_packets_arrived: int, packets_available_to_send: int) -> None:
        """
        真实更新：修改 self.buffer
        """
        new_buffer, sent, dropped = self._process_logic(
            self.buffer,  # 直接传入引用
            num_packets_arrived,
            packets_available_to_send
        )
        # 这一步因为 _process_logic 中用了 np.roll 可能会返回新数组，
        # 所以需要把值赋回去 (如果是原地修改则不需要)
        self.buffer = new_buffer
        self.sent_packets = sent
        self.dropped_packets = dropped

    def simulate_process(self, num_packets_arrived: int, packets_available_to_send: int):
        """
        模拟推演：只用于计算Reward，不修改自身状态
        """
        # 关键优化：只拷贝这个极小的 numpy 数组 (内存级拷贝，极快)
        temp_buffer = self.buffer.copy()

        new_buffer, sent, dropped = self._process_logic(
            temp_buffer,
            num_packets_arrived,
            packets_available_to_send
        )

        # 计算用于 Reward 的统计指标
        # 注意：这里需要复用 get_buffer_occupancy 和 get_avg_delay 的逻辑，但针对 temp_buffer
        occupancy = np.sum(new_buffer) / self.max_packets_buffer

        avg_delay = 0.0
        sum_pkts = np.sum(new_buffer)
        if sum_pkts != 0:
            avg_delay = float(
                np.sum(new_buffer * np.arange(self.max_packets_age + 1)) / sum_pkts
            )

        return {
            "sent_packets": sent,
            "dropped_packets": dropped,
            "buffer_occupancy": occupancy,
            "buffer_latency": avg_delay
        }

    def receive_packets(self, num_packets_arrived: int) -> None:
        # 1. 记录超时被丢弃的包
        self.dropped_packets = self.buffer[-1]

        # 2. 滚动缓冲区
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = 0  # 清零第一个位置

        # 3. 计算当前缓冲区占用量
        current_occupancy = np.sum(self.buffer)

        # 4. 计算剩余空间并处理新包
        available_space = self.max_packets_buffer - current_occupancy
        packets_to_receive = min(num_packets_arrived, available_space)

        # 5. 更新丢弃包数和缓冲区
        self.dropped_packets += num_packets_arrived - packets_to_receive
        self.buffer[0] = packets_to_receive

    def send_packets(self, packets_available_to_send: int) -> None:
        if packets_available_to_send == 0:
            self.sent_packets = 0
            return

        initial_total_packets = np.sum(self.buffer)
        if initial_total_packets == 0:
            self.sent_packets = 0
            return

        # 计算从右到左的累积和（降序）
        cumulative_sum_from_right = np.cumsum(self.buffer[::-1])[::-1]

        # 正确的搜索方法：手动查找或转换为升序
        k = len(self.buffer)
        for i in range(len(cumulative_sum_from_right)):
            if cumulative_sum_from_right[i] >= packets_available_to_send:
                k = i
                break

        if k < self.buffer.shape[0]:
            packets_at_k = packets_available_to_send
            if k + 1 < self.buffer.shape[0]:
                packets_at_k -= cumulative_sum_from_right[k + 1]

            self.buffer[k + 1:] = 0
            self.buffer[k] -= packets_at_k
        else:
            self.buffer[:] = 0

        final_total_packets = np.sum(self.buffer)
        self.sent_packets = initial_total_packets - final_total_packets

    def get_buffer_occupancy(self) -> float:
        """Get buffer occupancy rate.

        Returns
        -------
        float
            Buffer occupancy rate
        """
        return np.sum(self.buffer) / self.max_packets_buffer

    def get_avg_delay(self) -> float:
        """Get average buffer delay.

        Returns
        -------
        float
            Average buffer delay
        """
        if np.sum(self.buffer) != 0:
            return float(
                np.sum(self.buffer * np.arange(self.max_packets_age + 1))
                / np.sum(self.buffer)
            )
        else:
            return 0


class UEs:
    """
    完全向量化优化的 UEs 类。
    不再使用 Buffer 对象列表，而是直接管理二维状态矩阵。
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
            self,
            max_number_ues: int,
            max_buffer_latencies: np.ndarray,
            max_buffer_pkts: np.ndarray,
            pkt_sizes: np.ndarray,
    ) -> None:
        self.max_number_ues = max_number_ues
        self.max_buffer_latencies = max_buffer_latencies.astype(int)
        self.max_buffer_pkts = max_buffer_pkts
        self.pkt_sizes = pkt_sizes

        # === 向量化核心状态 ===
        # 找出所有用户中最大的时延窗口，统一矩阵大小
        # Shape: [N_UEs, Max_Window_Size]
        self.global_max_age = int(np.max(max_buffer_latencies)) + 1
        self.buffer_state = np.zeros((max_number_ues, self.global_max_age), dtype=np.float32)

        # 用于计算延迟的权重矩阵 [0, 1, 2, ..., max_age]
        self.delay_weights = np.arange(self.global_max_age, dtype=np.float32)

    def get_pkt_throughputs(
            self,
            sched_decision: np.ndarray,
            spectral_efficiencies: np.ndarray,
            bandwidth: float,
            num_available_rbs: int,
            pkt_sizes: np.ndarray,
    ) -> np.ndarray:
        # 保持原有的向量化计算
        # sched_decision: [N_UEs, N_RBs] (如果外部传入时维度正确)
        # 注意：这里假设输入维度已经对此齐，通常这部分已经是向量化的

        # 如果 sched_decision 是 [N_UEs, N_RBs]，spectral 是 [N_UEs, N_RBs]
        # sum(axis=1) 得到 [N_UEs]

        throughput_bits = np.sum(
            (bandwidth / num_available_rbs)
            * sched_decision
            * spectral_efficiencies,
            axis=1
        )
        return np.floor(throughput_bits / pkt_sizes)

    def _process_buffer_logic(
            self,
            buffer_matrix: np.ndarray,
            pkt_incomings: np.ndarray,
            pkt_throughputs: np.ndarray
    ):
        n_ues = self.max_number_ues

        # === [FIX START] 修正丢包逻辑：根据每个用户各自的 latency 丢包 ===

        # 1. 创建时间索引矩阵 [1, Max_Age] -> 广播成 [N_UEs, Max_Age]
        # 例如: [[0, 1, 2, ..., 100], [0, 1, 2, ..., 100]]
        age_indices = np.arange(self.global_max_age).reshape(1, -1)

        # 2. 获取每个用户的最大时延限制 [N_UEs, 1]
        # 注意：max_buffer_latencies 需要是列向量以便广播
        latency_limits = self.max_buffer_latencies.reshape(-1, 1)

        # 3. 生成过期掩码 (True 表示该位置的包已经超时)
        # 比如 UE_A limit=10, 那么 index >= 10 的位置都是 True
        expired_mask = age_indices >= latency_limits

        # 4. 计算即将被丢弃的包
        # 现在的 dropped_pkts 不仅仅是最后一列，而是所有位于“过期区域”的包的总和
        # 这一步也自然包含了 buffer_matrix[:, -1]，因为 -1 肯定大于任何 limit
        dropped_pkts = np.sum(buffer_matrix * expired_mask, axis=1)

        # 5. 清除矩阵中过期的包
        buffer_matrix[expired_mask] = 0.0

        # === [FIX END] ===

        # 2. 滚动缓冲区 (时间流逝)
        # 向右移一位：[t] -> [t+1]
        buffer_matrix[:, 1:] = buffer_matrix[:, :-1]
        buffer_matrix[:, 0] = 0

        # 3. 计算当前占用量
        current_occupancy = np.sum(buffer_matrix, axis=1)

        # 4. 接收新包
        available_space = self.max_buffer_pkts - current_occupancy
        # 限制接收量不能超过剩余空间 (溢出丢包)
        accepted_pkts = np.minimum(pkt_incomings, available_space)

        # 累加丢包 (这里是 Buffer Overflow 导致的丢包，加上之前 Timeout 导致的丢包)
        dropped_pkts += (pkt_incomings - accepted_pkts)

        # 将新包放入 index 0
        buffer_matrix[:, 0] += accepted_pkts

        # 5. 发送包逻辑 (向量化 FIFO)
        initial_total = np.sum(buffer_matrix, axis=1)
        remaining_to_send = pkt_throughputs.copy()

        # 倒序循环发送
        for col_idx in range(self.global_max_age - 1, -1, -1):
            col_packets = buffer_matrix[:, col_idx]

            # 优化：如果是 0 就跳过（mask 已经清零了过期包，所以这里更干净）
            # 但为了保持向量化形状一致，通常还是直接计算 minimum
            can_send = np.minimum(col_packets, remaining_to_send)

            buffer_matrix[:, col_idx] -= can_send
            remaining_to_send -= can_send

        final_total = np.sum(buffer_matrix, axis=1)
        sent_packets = initial_total - final_total

        return buffer_matrix, sent_packets, dropped_pkts

    def step(
            self,
            sched_decision: np.ndarray,
            traffics: np.ndarray,
            spectral_efficiencies: np.ndarray,
            bandwidths: np.ndarray,
            num_available_rbs: np.ndarray,
            update_state: bool = True,
    ) -> dict:

        # [MOVED] 1. 先计算 pkt_incomings，确保后续逻辑可用
        # 移除 np.floor，允许小数包
        # pkt_incomings = traffics / self.pkt_sizes
        pkt_incomings = np.floor(traffics / self.pkt_sizes)

        # [CRITICAL FIX] 强制展平为 1D 数组 (N_UEs,)
        # 这样能确保与 buffer_matrix 的行一一对应，防止广播错误
        pkt_incomings = pkt_incomings.flatten()

        # 2. 计算物理层吞吐量
        pkt_throughputs = np.zeros(self.max_number_ues)

        if len(sched_decision.shape) == 3:  # [B, U, R]
            for b in range(len(sched_decision)):
                # 累加每个基站提供的吞吐量
                pkt_throughputs += self.get_pkt_throughputs(
                    sched_decision[b],
                    spectral_efficiencies[b],
                    float(bandwidths[b]),
                    int(num_available_rbs[b]),
                    self.pkt_sizes
                )
        else:
            # 单基站情况
            pkt_throughputs = self.get_pkt_throughputs(
                sched_decision, spectral_efficiencies,
                float(bandwidths[0]), int(num_available_rbs[0]), self.pkt_sizes
            )

        # [CRITICAL FIX] 强制展平 pkt_throughputs
        pkt_throughputs = pkt_throughputs.flatten()

        # 3. 准备缓冲区矩阵
        if update_state:
            target_matrix = self.buffer_state  # 直接引用
        else:
            target_matrix = self.buffer_state.copy()  # 拷贝一份用于模拟

        # 4. 执行向量化逻辑 (Buffer更新)
        final_matrix, sent, dropped = self._process_buffer_logic(
            target_matrix, pkt_incomings, pkt_throughputs
        )
        # [CRITICAL FIX] 将 "包数量" 还原为 "比特数"
        # 这样 Environment 拿到的就是 bits，可以直接和 SLA Target (bits) 做比较
        throughput_bits = sent * self.pkt_sizes
        dropped_bits = dropped * self.pkt_sizes

        # # [DEBUG RESULT]
        # if debug_flag:
        #     print(f"  - [RESULT] Actual Sent Sum: {np.sum(sent):.2f}")

        # 5. 计算统计指标
        current_occupancy_abs = np.sum(final_matrix, axis=1)

        # 避免除以零
        buffer_occupancies = np.divide(
            current_occupancy_abs,
            self.max_buffer_pkts,
            out=np.zeros_like(current_occupancy_abs),
            where=self.max_buffer_pkts != 0
        )

        # 计算延迟 (加权平均)
        weighted_sum = np.sum(final_matrix * self.delay_weights, axis=1)
        buffer_latencies = np.divide(
            weighted_sum,
            current_occupancy_abs,
            out=np.zeros_like(weighted_sum),
            where=current_occupancy_abs != 0
        )

        return {
            "pkt_incoming": pkt_incomings,
            "pkt_throughputs": pkt_throughputs,
            "pkt_effective_thr": throughput_bits,  # sent,
            "buffer_occupancies": buffer_occupancies,
            "buffer_latencies": buffer_latencies,
            "dropped_pkts": dropped_bits # dropped,
        }

    def update_ues(
            self,
            ue_indexes: np.ndarray,
            max_buffer_latencies: np.ndarray,
            max_buffer_pkts: np.ndarray,
            pkt_sizes: np.ndarray,
    ) -> None:
        """
        更新UE参数。注意：如果在运行过程中 max_buffer_latencies 变大超过了初始化的 matrix 大小，
        这里可能需要 resize matrix，或者初始化时就给足够大的空间。
        """
        self.max_buffer_latencies[ue_indexes] = max_buffer_latencies
        self.max_buffer_pkts[ue_indexes] = max_buffer_pkts
        self.pkt_sizes[ue_indexes] = pkt_sizes

        # 重置被修改用户的 buffer 状态 (原逻辑是用新 Buffer 覆盖，即清空)
        # 或者你想保留状态？原逻辑是 `self.buffers[ue_index] = Buffer(...)` -> 清空
        self.buffer_state[ue_indexes, :] = 0


class Association(ABC):
    """
    Associations abstract class to implement dynamic basestations, slices and UEs associations.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation
    max_number_basestations : int
        Maximum number of basestations in the simulation
    max_number_slices: int
        Maximum number of supported slices

    Methods
    -------
    step(self, step_number: int, episode_number: int)
        Generate 2D positions for each UE in the simulation
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
        paths_cfg=None,
        workdir: str = None,
    ) -> None:
        """
        Parameters
        ----------
        ues: UEs
            UE class containing all UEs
        max_number_ues : int
            Maximum number of UEs in the simulation
        max_number_basestations : int
            Maximum number of basestations in the simulation
        max_number_slices: int
            Maximum number of supported slices
        """
        self.ues = ues
        self.max_number_ues = max_number_ues
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.rng = rng
        self.paths_cfg = paths_cfg
        self.workdir = workdir
        self.project_root = None

    @abstractmethod
    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Update Basestations, Slices and UEs associations

        Parameters
        ----------
        basestation_ue_assoc: np.ndarray
            Numpy array associating UEs to basestations with a form BxU,
            where represents the maximum number of UEs
        basestation_slice_assoc: np.ndarray
            Numpy array associating basestations to slices with a form BxS,
            where B is the maximum number of basestations and S is the
            maximum number of slices
        slice_ue_assoc: Optional[np.ndarray]
            UE association to slices
        slice_req: dict
            Dictionary contaning the slice requirements defined for each slice
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation

        Returns
        -------
        basestation_ue_assoc: np.ndarray
            New Numpy array associating UEs to basestations with a form BxU,
            where represents the maximum number of UEs
        basestation_slice_assoc: np.ndarray
            New numpy array associating basestations to slices with a form BxS,
            where B is the maximum number of basestations and S is the
            maximum number of slices
        slice_ue_assoc: Optional[np.ndarray]
            New UE association to slices to update the existent one
        slice_req: dict
            New dictionary contaning the slice requirements defined for each slice
        """
        return (
            basestation_ue_assoc,
            basestation_slice_assoc,
            slice_ue_assoc,
            slice_req,
        )

class Basestations:
    """
    Basestations class to implement basestations associations with slices
    and UEs

    ...

    Attributes
    ----------
    max_number_basestations : int
        Maximum number of basestations in the simulation
    max_number_slices: int
        Maximum number of supported slices
    slice_assoc: np.ndarray
        Numpy array associating basestations to slices with a form BxS,
        where B is the maximum number of basestations and S is the
        maximum number of slices
    ue_assoc: np.ndarray
        Numpy array associating UEs to basestations with a form BxU,
        where U represents the maximum number of UEs
    bandwidths: np.ndarray
        Numpy array with with the bandwidth value for each basestation
    carrier_frequencies: np.ndarray
        Numpy array with carrier frequencies values for each basestation
    num_available_rbs: np.ndarray
        Numpy array with the number of resource blocks available in
        each basestation

    Methods
    -------
    get_assoc(self)
        Method that return basestations associations with slices and UEs
    update_assoc(self, slice_assoc: Optional[np.ndarray] = None,
    ue_assoc: Optional[np.ndarray] = None)
        Update association of basestations with slices and UEs
    get_number_slices_per_basestation(self)
        Return a numpy array with the number of slices per basestation
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        slice_assoc: np.ndarray,
        ue_assoc: np.ndarray,
        bandwidths: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        max_number_basestations : int
            Maximum number of basestations in the simulation
        max_number_slices: int
            Maximum number of slices int he simulation
        slice_assoc: np.ndarray
            Numpy array associating basestations to slices with a form BxS,
            where B is the maximum number of basestations and S is the
            maximum number of slices
        ue_assoc: np.ndarray
            Numpy array associating UEs to basestations with a form BxU,
            where represents the maximum number of UEs
        bandwidths: np.ndarray
            Numpy array with with the bandwidth value for each basestation
        carrier_frequencies: np.ndarray
            Numpy array with carrier frequencies values for each basestation
        num_available_rbs: np.ndarray
            Numpy array with the number of resource blocks available in
            each basestation

        """
        self.slice_assoc = slice_assoc
        self.ue_assoc = ue_assoc
        self.bandwidths = bandwidths

    def get_assoc(self) -> np.ndarray:
        """Return slices and UEs association with basestations

        Returns
        -------
        numpy.ndarray
            An array containing slice and UEs associations
        """
        return np.array([self.slice_assoc, self.ue_assoc])

    def update_assoc(
        self,
        slice_assoc: Optional[np.ndarray] = None,
        ue_assoc: Optional[np.ndarray] = None,
    ) -> None:
        """Update associations of basestations with slices and UEs.

        Parameters
        ----------
        slice_assoc: Optional[np.ndarray]
            Optional slice association to update the existent one
        ue_assoc: Optional[np.ndarray]
            Optional UE association to update the existent one
        """
        self.slice_assoc = (
            slice_assoc if slice_assoc is not None else self.slice_assoc
        )
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def get_number_slices_per_basestation(self) -> np.ndarray:
        """Return the number of slices per basestation

        Returns
        -------
        numpy.ndarray
            An array containing the number of slices per basestation
        """
        return np.sum(self.slice_assoc, axis=1)

class Channel(ABC):
    """
    Channel abstract class to implement a channel generator to be used
    in the simulation.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation
    max_number_basestations : int
        Maximum number of basestations in the simulation
    num_available_rbs : np.ndarray
        Number of radio resource blocks available per basestation

    Methods
    -------
    def step(self, step_number: int, episode_number: int,
            mobilities: np.ndarray)
        Abstract method to define the allocation of radio resources for UEs
        based on the observation space
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        num_available_rbs: np.ndarray,
        paths_cfg=None,
        workdir: str = None,
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        max_number_basestations : int
            Maximum number of basestations in the simulation
        num_available_rbs : np.ndarray
            Number of radio resource blocks available per basestation
        """
        self.num_available_rbs = num_available_rbs
        self.paths_cfg = paths_cfg
        self.workdir = workdir
        self.project_root = None

    @abstractmethod
    def step(
        self,
        step_number: int,
        episode_number: int,
        rb_allocation: np.ndarray,
    ) -> np.ndarray:
        """Abstract function to generate channel values per UExRB.

        Parameters
        ----------
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation
        mobilities: np.ndarray
            Numpy array containing the positions of the UEs in the system
            with shape Ux2, where U represents the maximum number of UEs
            in the system and 2 represents a 2D coordinate of the UE in
            the scenario
        sched_decision : np.ndarray
            An array containing all radio resources allocation for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the evaluated basestation.

        Returns
        -------
        np.ndarray
            An array containing all spectral efficiency values for each UE
            in all basestations in the format BxUxR, where B is the number
            of basestations, U is the number of UEs, and R is the number
            of resource blocks available in the given basestation.
        """
        pass

class Metrics:
    """
    Metrics class to save/load metrics from simulations.
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        paths_cfg=None,
        workdir: str = None,
    ) -> None:

        self.paths_cfg = paths_cfg
        self.workdir = workdir
        self.metrics_hist = {
            "pkt_incoming": [],
            "pkt_throughputs": [],
            "pkt_effective_thr": [],
            "buffer_occupancies": [],
            "buffer_latencies": [],
            "dropped_pkts": [],
            "mobility": [],
            "spectral_efficiencies": [],
            "sched_decision": [],
            "target_cell_power": [],
            "basestation_ue_assoc": [],
            "basestation_slice_assoc": [],
            "slice_ue_assoc": [],
            "slice_req": [],
            "intent_drift": [],
            "slice_priority": [],
            "reward": [],
            "obs": [],
            "agent_action": [],
            "pkt_incoming_bits": []
        }

    def reset(self) -> None:
        """清空所有指标历史数据"""
        for key in self.metrics_hist.keys():
            self.metrics_hist[key] = []

    def step(self, hist: dict) -> None:
        """Append metric values to local variable metrics_hist

        Parameters
        ----------
        hist : dict
            Metric values obtained in the last simulation step
        """
        for metric in hist.keys():
            self.metrics_hist[metric].append(hist[metric])


    def save(self, *path_parts: str) -> None:
        """Save collected metric values to an external file"""
        base_path = Path(metrics_dir_path(self.workdir, self.paths_cfg))
        full_path = base_path / Path(*path_parts)

        # 一行代码同时创建目录并保存
        full_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(full_path, **self.metrics_hist)

class Mobility(ABC):
    """
    Mobility abstract class to implement a UEs mobilities.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation

    Methods
    -------
    step(self, step_number: int, episode_number: int)
        Generate 2D positions for each UE in the simulation
    """

    def __init__(
        self,
        max_number_ues: int,
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        """
        self.max_number_ues = max_number_ues

    @abstractmethod
    def step(self, step_number: int, episode_number: int) -> np.ndarray:
        """Generate UEs movement in the simulation.

        Parameters
        ----------
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation

        Returns
        -------
        mobilities: np.ndarray
            Numpy array containing the positions of the UEs in the system
            with shape Ux2, where U represents the maximum number of UEs
            in the system and 2 represents a 2D coordinate of the UE in
            the scenario
        """
        pass

class Slices:
    """
    Slices class to implement associations with basestations and UEs

    ...

    Attributes
    ----------
    max_number_slices: int
        Maximum number of supported slices
    max_number_ues: int
        Maximum number of UEs in the simulation
    ue_assoc: np.ndarray
        Numpy array associating UEs to basestations with a form SxU,
        where S and U represents the maximum number of slices and UEs
    requirements: dict
        Dictionary contaning the slice requirements defined for each slice

    Methods
    -------
    update_assoc(self, slice_assoc: Optional[np.ndarray] = None,
    ue_assoc: Optional[np.ndarray] = None)
        Update association of basestations with slices and UEs
    def update_slice_req(self, requirements: dict)
        Update slice requirements
    get_number_ue_per_slice(self)
        Return a numpy array with the number of UEs per slice
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        max_number_slices: int,
        max_number_ues: int,
        ue_assoc: np.ndarray,
        requirements: dict = {},
    ) -> None:
        """
        Parameters
        ----------
        max_number_slices : int
            Maximum number of slices in the simulation
        max_number_ues : int
            Maximum number of UEs in the simulation
        ue_assoc: np.ndarray
            Numpy array associating UEs to basestations with a form SxU,
            where S and U represents the maximum number of slices and UEs
        requirements: dict
            Dictionary contaning the slice requirements defined for each slice
        """
        self.max_number_slices = max_number_slices
        self.max_number_ues = max_number_ues
        self.ue_assoc = ue_assoc  # Matrix of |Slices|x|UEs|
        self.requirements = requirements

    def update_assoc(
        self,
        ue_assoc: Optional[np.ndarray] = None,
    ) -> None:
        """Update associations of slices with UEs.

        Parameters
        ----------
        ue_assoc: Optional[np.ndarray]
            Optional UE association to update the existent one
        """
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def update_slice_req(self, requirements: dict) -> None:
        """Update slices requirements.

        Parameters
        ----------
        requirements: dict
            Requirements to update the existent ones
        """
        self.requirements = requirements

    def get_number_ue_per_slice(self) -> np.ndarray:
        """Return the number of UEs per slice

        Returns
        -------
        numpy.ndarray
            An array containing the number of UEs per slice
        """
        return np.sum(self.ue_assoc, axis=1)

class Traffic(ABC):
    """
    Traffic abstract class to implement a UEs traffics.

    ...

    Attributes
    ----------
    max_number_ues : int
        Maximum number of UEs in the simulation

    Methods
    -------
    step(self, step_number: int, episode_number: int)
        Generate throughput traffic for each UE in the simulation
    """

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        max_number_ues: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        """
        Parameters
        ----------
        max_number_ues : int
            Maximum number of UEs in the simulation
        """
        self.max_number_ues = max_number_ues
        self.rng = rng

    @abstractmethod
    def step(
        self,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> np.ndarray:
        """Generate UEs traffic in the simulation.

        Parameters
        ----------
        step_number: int
            Step number in the simulation
        episode_number: int
            Episode number in the simulation

        Returns
        -------
        np.ndarray
            Numpy array containing the throughput traffic of the UEs in
            the system
        """
        pass

class MultSliceAssociation(Association):
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
        paths_cfg=None,
        workdir: str = None,
        generator_mode: bool = False,
        difficulty_config: dict = None,
    ) -> None:
        super().__init__(
            ues,
            max_number_ues,
            max_number_basestations,
            max_number_slices,
            rng,
            paths_cfg=paths_cfg,
            workdir=workdir,
        )
        # Generate Mode
        self.min_number_slices = 3
        self.generator_mode = generator_mode
        self.max_number_slices = 5
        self.maximum_number_scenarios = 200
        self.current_episode = -1
        self.slices_to_use = np.array([])

        self.difficulty_config = difficulty_config or {
            "traffic": 2,
            "sla_latency": 1.0,
            "sla_throughput": 1.0
        }

        self.slice_types = [
            "control_case_2",
            "monitoring_case_1",
            "robotic_surgery_case_1",
            "robotic_diagnosis",
            "medical_monitoring",
            "uav_app_case_1",
            "uav_control_non_vlos",
            "vr_gaming",
            "cloud_gaming",
            "video_streaming_4k",
        ]
        self.expectation_params = {
            "at_least": np.greater_equal,
            "at_most": np.less_equal,
            "exactly": np.equal,
            "greater": np.greater,
            "one_of": np.isin,
            "smaller": np.less,
        }
        self.slice_type_model = {
            "control_case_2": {
                "name": "control_case_2",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.999999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 50,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 100,  # ms
                    "message_size": 1 * 1024 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 5,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "monitoring_case_1": {
                "name": "monitoring_case_1",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "throughput",
                        "value": 10,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts,
                    "buffer_latency": 100,  # ms
                    "message_size": 1 * 1024 * 8,
                    "mobility": 72,  # Km/h
                    "traffic": 10,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "robotic_surgery_case_1": {
                "name": "robotic_surgery_case_1",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.9999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 20,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 30,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 40,  # ms
                    "message_size": 2000 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 30,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "robotic_diagnosis": {
                "name": "robotic_diagnosis",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 20,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 15,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 40,  # ms
                    "message_size": 80 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 15,  # Mbps,
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "medical_monitoring": {
                "name": "medical_monitoring",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.9999,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 100,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 10,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 200,  # ms
                    "message_size": 1000 * 8,
                    "mobility": 0,  # Km/h
                    "traffic": 10,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "uav_app_case_1": {
                "name": "uav_app_case_1",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "latency",
                        "value": 200,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par2": {
                        "name": "throughput",
                        "value": 100,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 400,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 30,  # Km/h
                    "traffic": 100,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 4,
                },
            },
            "uav_control_non_vlos": {
                "name": "uav_control_non_vlos",
                "priority": 1,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.99,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 140,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 20,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 300,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 30,  # Km/h
                    "traffic": 20,  # Mbps
                    "min_number_ues": 4,
                    "max_number_ues": 5,
                },
            },
            "vr_gaming": {
                "name": "vr_gaming",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "reliability",
                        "value": 99.99,
                        "unit": "rate",
                        "operator": self.expectation_params["at_least"],
                    },
                    "par2": {
                        "name": "latency",
                        "value": 10,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par3": {
                        "name": "throughput",
                        "value": 100,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 1000,  # pkts
                    "buffer_latency": 20,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 100,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 4,
                },
            },
            "cloud_gaming": {
                "name": "cloud_gaming",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "latency",
                        "value": 80,
                        "unit": "ms",
                        "operator": self.expectation_params["at_most"],
                    },
                    "par2": {
                        "name": "throughput",
                        "value": 50,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 160,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 50,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 5,
                },
            },
            "video_streaming_4k": {
                "name": "video_streaming_4k",
                "priority": 0,
                "parameters": {
                    "par1": {
                        "name": "throughput",
                        "value": 30,
                        "unit": "Mbps",
                        "operator": self.expectation_params["at_least"],
                    },
                },
                "ues": {
                    "buffer_size": 1024 * 10,  # pkts
                    "buffer_latency": 100,  # ms
                    "message_size": 8192 * 8,  # bits
                    "mobility": 0,  # Km/h
                    "traffic": 30,  # Mbps
                    "min_number_ues": 2,
                    "max_number_ues": 5,
                },
            },
        }




        # Read Mode
        self.association_file = None
        self.hist_slice_ue_assoc = None
        self.hist_slices_to_use = None
        self.hist_slice_req = None
        self.hist_basestation_slice_assoc = None
        self.hist_basestation_ue_assoc = None

    def step(
        self,
        basestation_ue_assoc: np.ndarray,
        basestation_slice_assoc: np.ndarray,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        step_number = step_number % 1000
        if self.generator_mode:
            if step_number == 0:
                number_slices = self.rng.integers(
                    low=self.min_number_slices,
                    high=self.max_number_slices,
                    endpoint=True,
                )
                self.slices_to_use = self.rng.choice(
                    np.arange(self.max_number_slices),
                    number_slices,
                    replace=False,
                )
                basestation_slice_assoc[0, self.slices_to_use] = 1
                slice_req = {
                    f"slice_{id}": {}
                    for id in np.arange(self.max_number_slices)
                }
                slice_req = self.slice_generator(slice_req, self.slices_to_use)
                ues_per_slices = np.array(
                    [
                        self.rng.integers(
                            slice_req[f"slice_{slice_idx}"]["ues"][
                                "min_number_ues"
                            ],
                            slice_req[f"slice_{slice_idx}"]["ues"][
                                "max_number_ues"
                            ],
                            1,
                            endpoint=True,
                        )
                        for slice_idx in self.slices_to_use
                    ]
                ).flatten()
                active_ues = np.array(
                    self.rng.choice(
                        (basestation_ue_assoc[0] == 0).nonzero()[0],
                        int(np.sum(ues_per_slices)),
                        replace=False,
                    )
                )
                used_ues = 0
                used_slices = 0
                for idx in self.slices_to_use:
                    if basestation_slice_assoc[0, idx] == 1:
                        slice_ue_assoc[
                            idx,
                            active_ues[
                                used_ues : used_ues
                                + ues_per_slices[used_slices]
                            ],
                        ] = 1
                        used_ues += ues_per_slices[used_slices]
                        used_slices += 1
                basestation_ue_assoc = np.array(
                    [np.sum(slice_ue_assoc, axis=0)]
                )

                self.update_ues(slice_ue_assoc, self.slices_to_use, slice_req)

            return (
                basestation_ue_assoc,
                basestation_slice_assoc,
                slice_ue_assoc,
                slice_req,
            )
        else:
            episode_to_use, condition = self.choose_episode(
                episode_number, self.current_episode
            )
            if condition:
                self.load_episode_data(episode_to_use)  # Update variables
                self.update_ues(
                    self.hist_slice_ue_assoc[step_number],
                    self.hist_slices_to_use[step_number],
                    self.hist_slice_req[step_number],
                )

            return (
                self.hist_basestation_ue_assoc[step_number],
                self.hist_basestation_slice_assoc[step_number],
                self.hist_slice_ue_assoc[step_number],
                self.hist_slice_req[step_number],
            )

    def choose_episode(
        self,
        episode_number: int,
        current_episode: int,
    ) -> Tuple[int, bool]:
        episode_to_use = episode_number % self.maximum_number_scenarios
        if episode_to_use != current_episode:
            return (episode_to_use, True)
        return (0, False)

    def slice_generator(
        self, slice_req: dict, slices_to_use: np.ndarray
    ) -> dict:
        slices_to_create = self.rng.choice(
            len(self.slice_types), len(slices_to_use), replace=False
        )

        for idx, slice in enumerate(slices_to_create):
            slice_req[f"slice_{slices_to_use[idx]}"] = self.slice_type_model[
                self.slice_types[slice]
            ]

        return slice_req

    def update_ues(
        self,
        slice_ue_assoc: np.ndarray,
        slices_to_use: np.ndarray,
        slice_req: dict,
    ) -> None:
        def slice_info(
            parameter: str, num_ues: int, slice_req: dict
        ) -> np.ndarray:
            return np.repeat(
                slice_req[f"slice_{slice}"]["ues"][parameter], num_ues
            )

        for slice in slices_to_use:
            slice_ues = (slice_ue_assoc[slice] == 1).nonzero()[0]
            self.ues.update_ues(
                slice_ues,
                slice_info("buffer_latency", len(slice_ues), slice_req),
                slice_info("buffer_size", len(slice_ues), slice_req),
                slice_info("message_size", len(slice_ues), slice_req),
            )

    def load_episode_data(self, episode_number: int):
        assoc_path = association_file_path(
            self.paths_cfg,
            episode_number,
            explicit_root=self.project_root,
        )
        with np.load(
                assoc_path,
                allow_pickle=True,
        ) as data:
            self.hist_slice_ue_assoc = data["hist_slice_ue_assoc"]
            self.hist_slices_to_use = data["hist_slices_to_use"]
            self.hist_slice_req = data["hist_slice_req"]
            self.hist_basestation_slice_assoc = data["hist_basestation_slice_assoc"]
            self.hist_basestation_ue_assoc = data["hist_basestation_ue_assoc"]
        self.current_episode = episode_number

    # # [新增] 核心方法：对单个 slice_req 字典进行难度缩放
    # def _apply_scaling(self, slice_req: dict) -> dict:
    #     # 使用 deepcopy 防止修改原始的 self.slice_type_model 模板（如果引用了它）
    #     scaled_req = deepcopy(slice_req)
    #
    #     factors = self.difficulty_config
    #
    #     for slice_key, slice_data in scaled_req.items():
    #         if not slice_data:
    #             continue
    #
    #         # 1. 修改 UE 的 Traffic (负载)
    #         if "ues" in slice_data and "traffic" in slice_data["ues"]:
    #             original_traffic = slice_data["ues"]["traffic"]
    #             slice_data["ues"]["traffic"] = original_traffic * factors.get("traffic", 1.0)
    #
    #         # 2. 修改 SLA 参数 (Parameters)
    #         if "parameters" in slice_data:
    #             for par_key, par_val in slice_data["parameters"].items():
    #                 param_name = par_val.get("name")
    #                 original_val = par_val.get("value")
    #
    #                 # 针对不同指标应用不同的缩放逻辑
    #                 if param_name == "latency":
    #                     # 时延越低越难，所以乘以系数 (例如 0.8)
    #                     par_val["value"] = original_val * factors.get("sla_latency", 1.0)
    #
    #                 elif param_name == "throughput":
    #                     # 吞吐越高越难，所以乘以系数 (例如 1.2)
    #                     par_val["value"] = original_val * factors.get("sla_throughput", 1.0)
    #
    #     return scaled_req
    #
    # # [修改] 在加载数据时注入修改逻辑
    # def load_episode_data(self, episode_number: int):
    #     with np.load(
    #             association_file_path(self.paths_cfg, episode_number),
    #             allow_pickle=True,
    #     ) as data:
    #         self.hist_slice_ue_assoc = data["hist_slice_ue_assoc"]
    #         self.hist_slices_to_use = data["hist_slices_to_use"]
    #
    #         # --- 关键修改点开始 ---
    #         raw_hist_slice_req = data["hist_slice_req"]
    #
    #         # 我们需要对历史数据中的每一帧(step)的 slice_req 进行处理
    #         # 假设 raw_hist_slice_req 是一个长度为 total_steps 的数组或列表
    #         processed_hist_slice_req = []
    #
    #         for step_req in raw_hist_slice_req:
    #             # 1. (可选) 先应用你之前提到的 override 逻辑，用代码中的新配置覆盖文件配置
    #             # current_req = self._override_slice_requirements(step_req)
    #             # 或者如果你只想基于文件修改，就直接用 step_req
    #             current_req = step_req
    #
    #             # 2. 应用难度系数
    #             scaled_req = self._apply_scaling(current_req)
    #             processed_hist_slice_req.append(scaled_req)
    #
    #         self.hist_slice_req = np.array(processed_hist_slice_req)
    #         # --- 关键修改点结束 ---
    #
    #         self.hist_basestation_slice_assoc = data["hist_basestation_slice_assoc"]
    #         self.hist_basestation_ue_assoc = data["hist_basestation_ue_assoc"]
    #
    #     self.current_episode = episode_number

    def _override_slice_requirements(self, file_slice_req: dict) -> dict:
        """
        Hot-Patching: 使用代码中 self.slice_type_model 定义的最新参数 (低流量、宽SLA)
        覆盖从文件读取的旧参数 (高流量、严SLA)。
        """
        new_slice_req = {}

        for slice_key, slice_data in file_slice_req.items():
            if not slice_data:  # 空切片
                new_slice_req[slice_key] = {}
                continue

            # 获取切片类型名称 (例如 'vr_gaming', 'uav_app_case_1')
            slice_type_name = slice_data.get('name')

            # 如果这个类型在我们新的配置表中存在，就用新的覆盖旧的
            if slice_type_name and slice_type_name in self.slice_type_model:
                # 使用 deepcopy 防止引用污染
                new_slice_req[slice_key] = deepcopy(self.slice_type_model[slice_type_name])
            else:
                # 如果没找到对应类型，就保留原样 (Fallback)
                new_slice_req[slice_key] = slice_data

        return new_slice_req

class MultSliceAssociationSeq(MultSliceAssociation):
    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not hasattr(v, 'read')}

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __init__(
        self,
        ues: UEs,
        max_number_ues: int,
        max_number_basestations: int,
        max_number_slices: int,
        rng: np.random.Generator = np.random.default_rng(),
        paths_cfg=None,
        workdir: str = None,
        generator_mode: bool = False,
    ) -> None:
        super().__init__(
            ues=ues,
            max_number_ues=max_number_ues,
            max_number_basestations=max_number_basestations,
            max_number_slices=max_number_slices,
            rng=rng,
            paths_cfg=paths_cfg,
            workdir=workdir,
            generator_mode=generator_mode,
        )
        self.channels_per_scenario = 100

    def choose_episode(
        self,
        episode_number: int,
        current_episode: int,
    ) -> Tuple[int, bool]:
        episode_to_use = episode_number // self.channels_per_scenario
        if episode_to_use != current_episode:
            return (episode_to_use, True)
        return (0, False)

class SimpleMobility(Mobility):
    def __init__(
        self,
        max_number_ues: int,
    ) -> None:
        super().__init__(max_number_ues)

    def step(self, step_number: int, episode_number: int) -> np.ndarray:
        return np.ones((self.max_number_ues, 2))


class QuadrigaChannel(Channel):
    def __init__(
            self,
            num_available_rbs: np.ndarray,
            paths_cfg=None,
            workdir: str = None,
    ) -> None:
        super().__init__(
            num_available_rbs,
            paths_cfg=paths_cfg,
            workdir=workdir,
        )
        # === 优化变量初始化 ===
        self.current_episode_number = -1

        # 缓存容器：用于存放当前场景的完整数据
        self.cached_channel_data = None
        self.cached_association_idx = -1
        self.cached_episode_idx = -1

        self.spectral_efficiencies = np.array([])
        self.transmission_power = 100  # Watts
        self.thermal_noise_power = 10e-14
        self.channel_eps_per_scenario = 100

    def _load_data_if_needed(self, episode_number: int):
        """
        辅助方法：检查缓存是否命中，未命中则加载数据到内存
        """
        association_to_use, episode_to_use, condition = self.choose_episode(
            episode_number, self.current_episode_number
        )

        # 如果环境发生变化 (condition=True) 或者缓存为空，加载数据
        if condition or self.cached_channel_data is None:
            self.current_episode_number = episode_number

            target_assoc = association_to_use if condition else self.cached_association_idx
            target_ep = episode_to_use if condition else self.cached_episode_idx

            if (target_assoc != self.cached_association_idx) or \
                    (target_ep != self.cached_episode_idx) or \
                    (self.cached_channel_data is None):

                npz_path = channel_npz_path(
                    self.paths_cfg,
                    target_assoc,
                    target_ep,
                    explicit_root=self.project_root,
                )
                mat_path = channel_mat_path(
                    self.paths_cfg,
                    target_assoc,
                    target_ep,
                    explicit_root=self.project_root,
                )

                try:
                    if os.path.exists(npz_path):
                        # fork-safe: pure numpy, supports SubprocVecEnv
                        with np.load(npz_path) as f:
                            self.cached_channel_data = f["target_cell_power"]
                    else:
                        # fallback: original h5py path (single-process only)
                        with h5py.File(mat_path, "r") as f:
                            self.cached_channel_data = f["target_cell_power"][:]

                    self.cached_association_idx = target_assoc
                    self.cached_episode_idx = target_ep

                except Exception as e:
                    ref = npz_path if os.path.exists(npz_path) else mat_path
                    raise RuntimeError(f"Failed to load channel file at '{ref}'. Error: {str(e)}")

    def step(
            self,
            step_number: int,
            episode_number: int,
            rb_allocation: np.ndarray,
    ) -> np.ndarray:
        step_number = step_number % 1000

        # 1. 确保数据已加载到内存
        self._load_data_if_needed(episode_number)

        # 2. 从内存读取 (极快)
        raw_slice = self.cached_channel_data[step_number]
        target_cell_power = np.squeeze(raw_slice)  # [135, 25] 或类似维度

        # 3. 物理层计算 (基于 rb_allocation)
        power_allocation = rb_allocation.T
        intercell_interference = np.zeros_like(target_cell_power)

        signal_power = power_allocation * target_cell_power
        sinr = signal_power / (intercell_interference + self.thermal_noise_power)
        spectral_efficiencies_per_rb = np.log2(1 + sinr)

        self.spectral_efficiencies = spectral_efficiencies_per_rb.T
        spectral_efficiencies = np.expand_dims(self.spectral_efficiencies, axis=0)
        refactorized_target_cell_power = target_cell_power.T

        return spectral_efficiencies, refactorized_target_cell_power

    def step_origin(
            self,
            step_number: int,
            episode_number: int,
    ) -> np.ndarray:
        """
        保留的方法：使用内存缓存优化后的旧版计算逻辑
        """
        step_number = step_number % 1000

        # 1. 确保数据已加载到内存 (复用相同的缓存逻辑)
        self._load_data_if_needed(episode_number)

        # 2. 从内存读取
        # 注意：这里不需要像 step 那样 squeeze 太多，保留维度以匹配原逻辑的 broadcast
        target_cell_power = self.cached_channel_data[step_number]  # [135, 1, 1, 25]

        # 3. 物理层计算 (原逻辑：基于平均功率分配)
        intercell_interference = np.zeros_like(target_cell_power)

        # 原逻辑计算公式
        power_factor = self.transmission_power / self.num_available_rbs[0]

        spectral_efficiencies_per_rb = np.log2(
            1
            + np.divide(
                power_factor * target_cell_power,
                (intercell_interference + self.thermal_noise_power),
            )
        )

        self.spectral_efficiencies = np.squeeze(
            spectral_efficiencies_per_rb.transpose()
        )

        return np.array([self.spectral_efficiencies])

    def choose_episode(
            self,
            episode_number: int,
            current_episode: int,
    ) -> Tuple[int, int, bool]:
        if episode_number != current_episode:
            association_to_use = episode_number
            episode_to_use = 0
            return (association_to_use, episode_to_use, True)
        return (0, 0, False)


class QuadrigaChannelSeq(QuadrigaChannel):
    def __init__(
            self,
            num_available_rbs: np.ndarray,
            paths_cfg=None,
            workdir: str = None,
    ) -> None:
        super().__init__(
            num_available_rbs,
            paths_cfg=paths_cfg,
            workdir=workdir,
        )

    def choose_episode(
            self,
            episode_number: int,
            current_episode: int,
    ) -> Tuple[int, int, bool]:
        if episode_number != current_episode:
            association_to_use = (
                    episode_number // self.channel_eps_per_scenario
            )
            episode_to_use = episode_number % self.channel_eps_per_scenario
            return (association_to_use, episode_to_use, True)
        return (0, 0, False)


class MultSliceTraffic(Traffic):
    def __init__(
        self,
        max_number_ues: int,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        super().__init__(max_number_ues, rng)

    def step(
        self,
        slice_ue_assoc: np.ndarray,
        slice_req: dict,
        step_number: int,
        episode_number: int,
    ) -> np.ndarray:
        traffic_per_ue = np.zeros(self.max_number_ues)
        # # [DEBUG START]
        # if step_number % 50 == 0:  # 限制打印频率
        #     print(f"\n[TRAFFIC DEBUG] Step {step_number}")
        #     print(f"  - Slice UE Assoc Shape: {slice_ue_assoc.shape}")
        #     print(f"  - Slice Req Keys: {list(slice_req.keys())}")
        # # [DEBUG END]

        for slice in slice_req:
            # 解析 Slice Index (例如 "slice_0" -> 0)
            try:
                s_idx = int(slice.split('_')[1])  # 更稳健的写法
            except:
                s_idx = int(slice[6])  # 原有写法

            # 获取该切片的用户索引
            idx_ues = (slice_ue_assoc[s_idx, :] == 1).nonzero()[0]

            if len(idx_ues) > 0:
                # 生成流量 (泊松分布)
                # 注意：这里生成的是 bits
                generated_val = self.rng.poisson(
                    slice_req[slice]["ues"]["traffic"], len(idx_ues)
                ) * 1e6

                traffic_per_ue[idx_ues] = generated_val

        return traffic_per_ue


