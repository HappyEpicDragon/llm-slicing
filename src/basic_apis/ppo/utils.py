from collections import deque
from typing import Tuple

import numpy as np

from .ppo_baseline.env_ray import MARLCommEnv


def get_metric_value(
        metric_name: str,
        last_unformatted_obs: deque,
        slice_idx: int,
        slice_ues: np.ndarray,
        reliability_pkt_loss: bool = False,
) -> np.ndarray:
    """
    提取原始指标值，不做任何逻辑放宽或修改。
    逻辑判断全部移交到 intent_drift_calc 中处理。
    """

    def calc_metric_interval(metric: str, slice_ues: np.ndarray) -> float:
        return np.sum(
            [
                last_unformatted_obs[i][metric][slice_ues]
                for i in range(len(last_unformatted_obs))
            ],
            axis=0,
        )

    if metric_name == "throughput":
        # pkt_effective_thr 已是 bit/step，直接转 Mbps/step，避免重复乘 message_size
        metric_value = last_unformatted_obs[0]["pkt_effective_thr"][slice_ues] / 1e6

    # if metric_name == "throughput":
    #     # [OLD] 只看最新一帧
    #     # metric_value = (last_unformatted_obs[0]["pkt_effective_thr"][slice_ues] * ...)
    #
    #     # [NEW] 计算 Deque 中所有帧的平均值 (平滑处理)
    #     # 假设 deque 长度为 N (通常是 10 或更多)
    #     total_thr_sum = 0.0
    #     valid_steps = 0
    #
    #     for obs in last_unformatted_obs:
    #         if "pkt_effective_thr" in obs:
    #             total_thr_sum += obs["pkt_effective_thr"][slice_ues]
    #             valid_steps += 1
    #
    #     if valid_steps > 0:
    #         avg_thr = total_thr_sum / valid_steps
    #     else:
    #         avg_thr = 0.0
    #
    #     # 转换为 Mbps
    #     metric_value = (
    #                            avg_thr
    #                            * last_unformatted_obs[0]["slice_req"][f"slice_{slice_idx}"]["ues"]["message_size"]
    #                    ) / 1e6

    elif metric_name == "reliability":
        if reliability_pkt_loss:
            pkts_snt_over_interval = calc_metric_interval("pkt_effective_thr", slice_ues)
            dropped_pkts_over_interval = calc_metric_interval("dropped_pkts", slice_ues)
            buffer_pkts = (
                    last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
                    * last_unformatted_obs[0]["slice_req"][f"slice_{slice_idx}"]["ues"]["buffer_size"]
                    + dropped_pkts_over_interval
                    + pkts_snt_over_interval
            )
            # 避免除以零
            metric_value = np.divide(
                dropped_pkts_over_interval,
                buffer_pkts,
                where=buffer_pkts != 0,
                out=np.zeros_like(buffer_pkts, dtype=float),
            )  # Rate [0,1] (Packet Loss Rate)
        else:
            metric_value = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]  # Rate [0,1]

    elif metric_name == "latency":
        metric_value = last_unformatted_obs[0]["buffer_latencies"][slice_ues]  # Seconds

    else:
        raise ValueError(f"Invalid metric name: {metric_name}")

    return metric_value


def intent_drift_calc(
    last_unformatted_obs: deque[dict],
    max_number_ues_slice: int,
    intent_overfulfillment_rate: float,
    reliability_pkt_loss: bool = True,
) -> np.ndarray:
    last_obs_slice_req = last_unformatted_obs[0]["slice_req"]
    metrics = {"throughput": 0, "reliability": 1, "latency": 2}
    observations = np.zeros(
        (
            last_unformatted_obs[0]["slice_ue_assoc"].shape[0],
            max_number_ues_slice,
            len(metrics),
        ),
        dtype=float,
    )

    for slice in last_obs_slice_req:
        if last_obs_slice_req[slice] == {}:
            continue
        slice_idx = int(slice.split("_")[1])
        slice_ues = last_unformatted_obs[0]["slice_ue_assoc"][
            slice_idx
        ].nonzero()[0]
        for parameter in last_obs_slice_req[slice]["parameters"].values():
            metric_value = get_metric_value(
                parameter["name"],
                last_unformatted_obs,
                slice_idx,
                slice_ues,
                reliability_pkt_loss,
            )
            if parameter["name"] == "throughput":
                # pkt_effective_thr 已是 bit/step，直接转 Mbps/step
                metric_value = last_unformatted_obs[0]["pkt_effective_thr"][slice_ues] / 1e6

                buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
                if len(last_unformatted_obs) > 1:
                    prev_buffer_occ = last_unformatted_obs[1]["buffer_occupancies"][slice_ues]
                else:
                    prev_buffer_occ = buffer_occ.copy()

                for idx in range(len(slice_ues)):
                    if (metric_value[idx] < parameter["value"]) and (
                            (buffer_occ[idx] < 0.02) or (prev_buffer_occ[idx] < 0.02)
                    ):
                        metric_value[idx] = parameter["value"] * (1.1 + intent_overfulfillment_rate)

                # if np.random.rand() < 0.01:
                #     print(
                #         f"[THR DEBUG] Slice {slice_idx} mean Mbps={np.mean(metric_value):.2f}, "
                #         f"target={parameter['value']}"
                #     )


            buffer_occupancy_threshold = 0.6

            if parameter["name"] == "reliability":
                if reliability_pkt_loss:
                    intent_fulfillment = parameter["operator"](
                        100 * (1 - metric_value), parameter["value"]
                    ).astype(int)
                else:
                    intent_fulfillment = parameter["operator"](
                        1 - metric_value, (1 - buffer_occupancy_threshold)
                    ).astype(int)
            else:
                intent_fulfillment = parameter["operator"](
                    metric_value, parameter["value"]
                ).astype(int)
            intent_unfulfillment = np.logical_not(intent_fulfillment)

            match parameter["name"]:
                case "throughput":
                    # Intent fulfillment
                    if np.sum(intent_fulfillment) > 0:
                        overfulfilled_mask = intent_fulfillment * (
                            metric_value
                            > (
                                parameter["value"]
                                * (1 + intent_overfulfillment_rate)
                            )
                        )
                        fulfilled_mask = (
                            intent_fulfillment
                            * np.logical_not(overfulfilled_mask)
                        ).nonzero()[0]
                        overfulfilled_mask = overfulfilled_mask.nonzero()[0]
                        # Fulfilled intent
                        observations[
                            slice_idx,
                            fulfilled_mask,
                            metrics[parameter["name"]],
                        ] += (
                            metric_value[fulfilled_mask] - parameter["value"]
                        ) / (
                            parameter["value"] * intent_overfulfillment_rate
                        )
                        # Overfulfilled intent
                        observations[
                            slice_idx,
                            overfulfilled_mask,
                            metrics[parameter["name"]],
                        ] += 1

                    # Intent unfulfillment
                    if np.sum(intent_unfulfillment) > 0:
                        observations[
                            slice_idx,
                            intent_unfulfillment.nonzero()[0],
                            metrics[parameter["name"]],
                        ] -= (
                            parameter["value"]
                            - metric_value[intent_unfulfillment.nonzero()[0]]
                        ) / (
                            parameter["value"]
                        )

                case "reliability":
                    if reliability_pkt_loss:
                        # Using pkt loss
                        # Intent fulfillment
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value
                                < (
                                    ((100 - parameter["value"]) / 100)
                                    * (1 - intent_overfulfillment_rate)
                                )
                            )
                            fulfilled_mask = (
                                intent_fulfillment
                                * np.logical_not(overfulfilled_mask)
                            ).nonzero()[0]

                            overfulfilled_mask = overfulfilled_mask.nonzero()[
                                0
                            ]
                            # Fulfilled intent
                            observations[
                                slice_idx,
                                fulfilled_mask,
                                metrics[parameter["name"]],
                            ] += (
                                (100 - parameter["value"]) / 100
                                - metric_value[fulfilled_mask]
                            ) / (
                                ((100 - parameter["value"]) / 100)
                                * intent_overfulfillment_rate
                            )
                            # Overfulfilled intent
                            observations[
                                slice_idx,
                                overfulfilled_mask,
                                metrics[parameter["name"]],
                            ] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx,
                                intent_unfulfillment.nonzero()[0],
                                metrics[parameter["name"]],
                            ] -= (
                                metric_value[intent_unfulfillment.nonzero()[0]]
                                - ((100 - parameter["value"]) / 100)
                            ) / (
                                parameter["value"] / 100
                            )

                    else:
                        # Using buffer occupancy instead of packet loss
                        # Intent fulfillment
                        buffer_occupancy_over_threshold = 0.2
                        if np.sum(intent_fulfillment) > 0:
                            overfulfilled_mask = intent_fulfillment * (
                                metric_value <= buffer_occupancy_over_threshold
                            )
                            fulfilled_mask = (
                                intent_fulfillment
                                * np.logical_not(overfulfilled_mask)
                            ).nonzero()[0]

                            overfulfilled_mask = overfulfilled_mask.nonzero()[
                                0
                            ]
                            # Fulfilled intent
                            observations[
                                slice_idx,
                                fulfilled_mask,
                                metrics[parameter["name"]],
                            ] += (
                                buffer_occupancy_threshold
                                - metric_value[fulfilled_mask]
                            ) / (
                                buffer_occupancy_threshold
                                - buffer_occupancy_over_threshold
                            )
                            # Overfulfilled intent
                            observations[
                                slice_idx,
                                overfulfilled_mask,
                                metrics[parameter["name"]],
                            ] += 1

                        # Intent unfulfillment
                        if np.sum(intent_unfulfillment) > 0:
                            observations[
                                slice_idx,
                                intent_unfulfillment.nonzero()[0],
                                metrics[parameter["name"]],
                            ] -= (
                                metric_value[intent_unfulfillment.nonzero()[0]]
                                - buffer_occupancy_threshold
                            ) / (
                                1 - buffer_occupancy_threshold
                            )

                case "latency":
                    max_latency_per_ue = (
                        np.ones_like(slice_ues)
                        * last_obs_slice_req[slice]["ues"]["buffer_latency"]
                    )
                    # Intent fulfillment
                    if np.sum(intent_fulfillment) > 0:
                        overfulfilled_mask = intent_fulfillment * (
                            metric_value
                            < (
                                parameter["value"]
                                * (1 - intent_overfulfillment_rate)
                            )
                        )
                        fulfilled_mask = (
                            intent_fulfillment
                            * np.logical_not(overfulfilled_mask)
                        ).nonzero()[0]
                        overfulfilled_mask = (
                            intent_fulfillment * overfulfilled_mask
                        ).nonzero()[0]
                        # Fulfilled intent
                        observations[
                            slice_idx,
                            fulfilled_mask,
                            metrics[parameter["name"]],
                        ] += (
                            parameter["value"] - metric_value[fulfilled_mask]
                        ) / (
                            parameter["value"] * intent_overfulfillment_rate
                        )
                        # Overfulfilled intent
                        observations[
                            slice_idx,
                            overfulfilled_mask,
                            metrics[parameter["name"]],
                        ] += 1

                    # Intent unfulfillment
                    if np.sum(intent_unfulfillment) > 0:
                        observations[
                            slice_idx,
                            intent_unfulfillment.nonzero()[0],
                            metrics[parameter["name"]],
                        ] -= (
                            metric_value[intent_unfulfillment.nonzero()[0]]
                            - parameter["value"]
                        ) / (
                            max_latency_per_ue[
                                intent_unfulfillment.nonzero()[0]
                            ]
                            - parameter["value"]
                        )

                case _:
                    raise ValueError("Invalid parameter name")

    return observations

# def intent_drift_calc(
#         last_unformatted_obs: deque,
#         max_number_ues_slice: int,
#         intent_overfulfillment_rate: float,
#         reliability_pkt_loss: bool = True,
# ) -> np.ndarray:
#     """
#     计算 SLA 偏移 (Intent Drift)。
#     修改说明：
#     1. 修复了 Throughput 逻辑漏洞：严格检查是否存在 Traffic Demand。
#     2. 只有在有需求（Buffer>0 或 Incoming>0）时，才计算 Throughput 违约。
#     3. 如果没有需求，Throughput Drift 设为 0 (达标/无影响)。
#     """
#     current_obs = last_unformatted_obs[0]
#     last_obs_slice_req = current_obs["slice_req"]
#
#     metrics_map = {"throughput": 0, "reliability": 1, "latency": 2}
#
#     # 初始化 Drift 矩阵: [Slices, Max_UEs, Metrics]
#     observations = np.zeros(
#         (
#             current_obs["slice_ue_assoc"].shape[0],
#             max_number_ues_slice,
#             len(metrics_map),
#         ),
#         dtype=float,
#     )
#
#     for slice_key in last_obs_slice_req:
#         if not last_obs_slice_req[slice_key]:
#             continue
#
#         # 解析 Slice ID
#         try:
#             slice_idx = int(slice_key.split("_")[1])
#         except Exception:
#             # Fallback for single digit cases if name format differs
#             slice_idx = int(slice_key[-1])
#
#         slice_ues = current_obs["slice_ue_assoc"][slice_idx].nonzero()[0]
#         if len(slice_ues) == 0:
#             continue
#
#         for parameter in last_obs_slice_req[slice_key]["parameters"].values():
#             metric_name = parameter["name"]
#             target_value = parameter["value"]
#             metric_idx = metrics_map[metric_name]
#
#             # 获取原始指标值
#             metric_values = get_metric_value(
#                 metric_name,
#                 last_unformatted_obs,
#                 slice_idx,
#                 slice_ues,
#                 reliability_pkt_loss,
#             )
#
#             # =========================================================
#             # 1. Throughput 处理 (逻辑修复核心)
#             # =========================================================
#             if metric_name == "throughput":
#                 # 获取 Buffer 占用和新到达流量
#                 buffer_occ = current_obs["buffer_occupancies"][slice_ues]
#
#                 # 尝试获取 pkt_incoming_bits (One-Shot Env 新增)，如果没有则回退到 pkt_incoming
#                 if "pkt_incoming_bits" in current_obs:
#                     # 转换为 Mbps 估计值用于判断是否有流量
#                     incoming_load = current_obs["pkt_incoming_bits"][slice_ues]
#                 elif "pkt_incoming" in current_obs:
#                     incoming_load = current_obs["pkt_incoming"][slice_ues]
#                 else:
#                     incoming_load = np.zeros_like(buffer_occ)
#
#                 # 判定是否存在需求：Buffer 不为空 或 有新流量到达
#                 # 使用小阈值避免浮点误差
#                 has_demand = (buffer_occ > 1e-5) | (incoming_load > 1e-5)
#
#                 # 计算 Fulfillment Mask
#                 # 只有在有需求的情况下，小于 Target 才算未达标
#                 # 如果没有需求 (has_demand=False)，则认为达标 (metric_values 此时通常为0)
#
#                 # fulfilled: (Val >= Target) OR (No Demand)
#                 is_fulfilled = (metric_values >= target_value) | (~has_demand)
#
#                 # overfulfilled: (Val > Target * 1.2) AND (Has Demand)
#                 # 只有在真的有传输且超量时才算 Overfulfilled
#                 is_overfulfilled = (metric_values > target_value * (1 + intent_overfulfillment_rate)) & has_demand
#
#                 # 修正 fulfilled，排除 overfulfilled
#                 is_fulfilled = is_fulfilled & (~is_overfulfilled)
#
#                 is_unfulfilled = ~ (is_fulfilled | is_overfulfilled)
#
#                 # --- 计算 Drift ---
#                 # 1. Fulfilled (0 ~ 1)
#                 if np.any(is_fulfilled):
#                     mask = is_fulfilled
#                     # 如果是因为没需求而 fulfilled，drift = 0
#                     # 如果是因为达标，计算归一化 drift
#
#                     # 这里的计算稍微 tricky：如果没有需求，metric_value 是 0，target 是正数。
#                     # 我们直接赋 0。如果有需求且达标，计算比例。
#
#                     drift_val = np.zeros_like(metric_values)
#                     # 仅对有需求且达标的部分计算正向 Drift
#                     valid_calc = mask & has_demand
#                     drift_val[valid_calc] = (metric_values[valid_calc] - target_value) / (
#                                 target_value * intent_overfulfillment_rate)
#
#                     observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift_val[mask]
#
#                 # 2. Overfulfilled (1)
#                 if np.any(is_overfulfilled):
#                     observations[slice_idx, :len(slice_ues)][is_overfulfilled, metric_idx] = 1.0
#
#                 # 3. Unfulfilled (-inf ~ 0) -> 真正的惩罚
#                 if np.any(is_unfulfilled):
#                     mask = is_unfulfilled
#                     # 违约程度：(Target - Actual) / Target
#                     drift_val = -(target_value - metric_values[mask]) / target_value
#                     observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift_val
#
#             # =========================================================
#             # 2. Reliability 处理
#             # =========================================================
#             elif metric_name == "reliability":
#                 if reliability_pkt_loss:
#                     # Packet Loss Rate based
#                     # value 是 reliability rate (e.g., 99.99), 转换成 loss rate threshold
#                     loss_threshold = (100 - target_value) / 100.0
#
#                     is_overfulfilled = metric_values < (loss_threshold * (1 - intent_overfulfillment_rate))
#                     is_fulfilled = (metric_values <= loss_threshold) & (~is_overfulfilled)
#                     is_unfulfilled = metric_values > loss_threshold
#
#                     if np.any(is_fulfilled):
#                         mask = is_fulfilled
#                         # 越小越好： (Threshold - Actual) / Range
#                         denominator = loss_threshold * intent_overfulfillment_rate
#                         # 避免除零
#                         if denominator == 0: denominator = 1e-6
#                         drift = (loss_threshold - metric_values[mask]) / denominator
#                         observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift
#
#                     if np.any(is_overfulfilled):
#                         observations[slice_idx, :len(slice_ues)][is_overfulfilled, metric_idx] = 1.0
#
#                     if np.any(is_unfulfilled):
#                         mask = is_unfulfilled
#                         drift = -(metric_values[mask] - loss_threshold) / loss_threshold
#                         observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift
#                 else:
#                     # Buffer Occupancy based (Fallback)
#                     buffer_threshold = 0.6
#                     buffer_over_threshold = 0.2
#
#                     is_overfulfilled = metric_values <= buffer_over_threshold
#                     is_fulfilled = (metric_values <= buffer_threshold) & (~is_overfulfilled)
#                     is_unfulfilled = metric_values > buffer_threshold
#
#                     if np.any(is_fulfilled):
#                         mask = is_fulfilled
#                         drift = (buffer_threshold - metric_values[mask]) / (buffer_threshold - buffer_over_threshold)
#                         observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift
#
#                     if np.any(is_overfulfilled):
#                         observations[slice_idx, :len(slice_ues)][is_overfulfilled, metric_idx] = 1.0
#
#                     if np.any(is_unfulfilled):
#                         mask = is_unfulfilled
#                         drift = -(metric_values[mask] - buffer_threshold) / (1 - buffer_threshold)
#                         observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift
#
#             # =========================================================
#             # 3. Latency 处理
#             # =========================================================
#             elif metric_name == "latency":
#                 # Target value is in ms (e.g., 10, 20)
#                 # metric_value is in Seconds? Check original code.
#                 # Assuming metric_value is adjusted to match target unit.
#                 # Original code: get_metric_value returns seconds for latency.
#                 # Parameter value usually ms. Let's Normalize.
#
#                 # IMPORTANT: Check units in Env. Usually Env returns ms in buffer_latencies?
#                 # Looking at `get_metric_value`: `metric_value = last_unformatted_obs[0]["buffer_latencies"]`
#                 # In `UEs` class, latencies usually calculated relative to TTI.
#                 # Let's assume units are consistent (e.g. all ms).
#                 # If metric_value is seconds and target is ms, this breaks.
#                 # *Fix*: Convert to ms if needed, but assuming raw data is consistent for now.
#
#                 max_latency_allowed = last_obs_slice_req[slice_key]["ues"]["buffer_latency"]  # ms
#
#                 is_overfulfilled = metric_values < (target_value * (1 - intent_overfulfillment_rate))
#                 is_fulfilled = (metric_values <= target_value) & (~is_overfulfilled)
#                 is_unfulfilled = metric_values > target_value
#
#                 if np.any(is_fulfilled):
#                     mask = is_fulfilled
#                     drift = (target_value - metric_values[mask]) / (target_value * intent_overfulfillment_rate)
#                     observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift
#
#                 if np.any(is_overfulfilled):
#                     observations[slice_idx, :len(slice_ues)][is_overfulfilled, metric_idx] = 1.0
#
#                 if np.any(is_unfulfilled):
#                     mask = is_unfulfilled
#                     denom = max_latency_allowed - target_value
#                     if denom <= 0: denom = 1e-6
#                     drift = -(metric_values[mask] - target_value) / denom
#                     observations[slice_idx, :len(slice_ues)][mask, metric_idx] += drift
#
#     return observations


def calculate_slice_ue_obs(
        max_number_ues_slice: int,
        intent_drift: np.ndarray,
        slice_idx: int,
        slice_ues: np.ndarray,
        slice_req: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    辅助函数：聚合每个 Slice 的观测信息
    """
    metrics_map = {"throughput": 0, "reliability": 1, "latency": 2}

    # 默认为 -2 (无效值标记)
    intent_ue_values = -2 * np.ones((max_number_ues_slice, len(metrics_map)))
    intent_slice_values = -2 * np.ones(len(metrics_map))

    if slice_ues.shape[0] > 0 and f"slice_{slice_idx}" in slice_req:
        req = slice_req[f"slice_{slice_idx}"]
        if "parameters" in req:
            # 获取该 Slice 关心的 metric indices
            active_metrics = [metrics_map[p["name"]] for p in req["parameters"].values()]

            for m_idx in active_metrics:
                # 提取 UE 级别的 drift
                # 限制数量不超过 max_number_ues_slice
                num_ues = min(slice_ues.shape[0], max_number_ues_slice)
                intent_ue_values[:num_ues, m_idx] = intent_drift[slice_idx, :num_ues, m_idx]

                # 计算 Slice 级别的平均 drift
                intent_slice_values[m_idx] = np.mean(intent_drift[slice_idx, :num_ues, m_idx])

    return (intent_ue_values, intent_slice_values)


# def calculate_reward_no_mask(
#     obs_space: dict,
#     last_formatted_obs: dict,
#     last_unformatted_obs: deque,
#     var_obs_per_slice: int,
#     priority_flag: bool = True,
# ) -> dict:
#     reward = {}
#     for player_idx, agent_obs in enumerate(last_formatted_obs.items()):
#         if player_idx == 0:
#             elements_idx = last_unformatted_obs[0]["basestation_slice_assoc"][
#                 0, :
#             ].nonzero()[0]
#             active_observations = np.zeros(len(last_formatted_obs) - 1)
#             slice_priorities = np.zeros(len(last_formatted_obs) - 1)
#             for element_idx in elements_idx:
#                 slice_priorities[element_idx] = last_unformatted_obs[0][
#                     "slice_req"
#                 ][f"slice_{element_idx}"]["priority"]
#                 metrics = agent_obs[1]["observations"][
#                     (element_idx * var_obs_per_slice) : (
#                         element_idx * var_obs_per_slice
#                     )
#                     + 3
#                 ]
#                 metrics = metrics[np.logical_not(np.isclose(metrics, -2))]
#                 metrics = np.min(metrics) if metrics.shape[0] > 0 else 1
#                 active_observations[element_idx] = metrics
#             if np.isclose(np.sum(active_observations < 0), 0):
#                 reward[agent_obs[0]] = np.mean(active_observations)
#             elif (
#                 not np.isclose(
#                     np.sum((slice_priorities * active_observations) < 0), 0
#                 )
#                 and priority_flag
#             ):
#                 negative_obs_idx = (
#                     active_observations * slice_priorities < 0
#                 ).nonzero()[0]
#                 reward[agent_obs[0]] = (
#                     np.mean(active_observations[negative_obs_idx]) - 1
#                 )
#             else:
#                 negative_obs_idx = (active_observations < 0).nonzero()[0]
#                 reward[agent_obs[0]] = np.mean(
#                     active_observations[negative_obs_idx]
#                 )
#         else:
#             reward[agent_obs[0]] = 0
#             active_metrics = agent_obs[1]["observations"][3:6]
#             if np.sum(active_metrics > 0) > 0:
#                 intent_drifts = agent_obs[1]["observations"][0:3]
#                 # intent_drifts[intent_drifts > 0] = 0
#                 active_metrics = agent_obs[1]["observations"][3:6]
#                 reward[agent_obs[0]] = np.min(
#                     intent_drifts[active_metrics.astype(bool)]
#                 )
#
#     return reward

def calculate_reward_no_mask(
        obs_space: dict,
        last_formatted_obs: dict,
        last_unformatted_obs: deque,
        var_obs_per_slice: int,
        priority_flag: bool = True,
) -> dict:
    """
    计算基础奖励 (Base Reward / SLA Penalty)。
    返回值是一个字典 {agent_id: reward}。
    """
    reward = {}

    # 假设是单智能体或名为 player_0 的主智能体
    # 这里逻辑主要处理 player_0

    for player_id, agent_data in last_formatted_obs.items():
        if player_id == "player_0":
            current_obs = last_unformatted_obs[0]

            # 获取所有激活切片的索引
            active_slice_indices = np.where(current_obs["basestation_slice_assoc"][0, :] == 1)[0]

            if len(active_slice_indices) == 0:
                reward[player_id] = 0.0
                continue

            slice_scores = []
            slice_priorities = []

            for s_idx in active_slice_indices:
                # 获取优先级
                prio = current_obs["slice_req"].get(f"slice_{s_idx}", {}).get("priority", 0)
                slice_priorities.append(prio)

                # 从 agent observation 中提取该切片的指标
                # 假设 obs 是扁平化的 [Slice0_Metrics, Slice1_Metrics...]
                start_idx = s_idx * var_obs_per_slice
                end_idx = start_idx + 3  # 3 metrics

                metrics = agent_data["observations"][start_idx:end_idx]

                # 过滤掉无效值 (-2)
                valid_metrics = metrics[metrics > -1.5]

                if len(valid_metrics) > 0:
                    # 取该切片最差的指标作为该切片的得分 (木桶效应)
                    min_metric = np.min(valid_metrics)
                    slice_scores.append(min_metric)
                else:
                    # 如果没有有效指标，认为是满分 (1.0) 或者中性 (0.0)
                    # 这里设为 0.0 (不惩罚也不奖励)
                    slice_scores.append(0.0)

            slice_scores = np.array(slice_scores)
            slice_priorities = np.array(slice_priorities)

            # --- 聚合所有切片的得分 ---

            # 1. 如果所有切片都非负 (没有严重违约)
            if np.all(slice_scores >= 0):
                reward[player_id] = np.mean(slice_scores)

            # 2. 如果有违约 (负分)
            else:
                if priority_flag:
                    # 只关注高优先级切片的违约情况
                    weighted_scores = slice_scores * slice_priorities
                    bad_prio_indices = np.where(weighted_scores < 0)[0]

                    if len(bad_prio_indices) > 0:
                        # 有高优先级切片违约 -> 严厉惩罚
                        # 惩罚基础值 -1，再加上违约程度
                        reward[player_id] = np.mean(slice_scores[bad_prio_indices]) - 1.0
                    else:
                        # 只有低优先级切片违约 -> 普通惩罚
                        bad_indices = np.where(slice_scores < 0)[0]
                        reward[player_id] = np.mean(slice_scores[bad_indices])
                else:
                    # 不区分优先级，计算所有违约切片的平均值
                    bad_indices = np.where(slice_scores < 0)[0]
                    reward[player_id] = np.mean(slice_scores[bad_indices])

        else:
            # 其他 Player (如果有) 暂时给 0
            reward[player_id] = 0.0

    return reward


def scores_to_rbs(
    action: np.ndarray, total_rbs: int, association: np.ndarray
) -> np.ndarray:
    rbs_per_unit = (
        round_int_equal_sum(
            total_rbs * (action + 1) / np.sum(action + 1),
            total_rbs,
        )
        if np.sum(action + 1) != 0
        else round_int_equal_sum(
            (total_rbs / np.sum(association)) * association,
            total_rbs,
        )
    )
    assert np.sum(rbs_per_unit < 0) == 0, "Negative RBs"
    assert (
        np.sum(rbs_per_unit * association).astype(int) == total_rbs
    ), f"Allocated RBs {np.sum(rbs_per_unit * association)} are different from available RBs {total_rbs}\n{action}\n{rbs_per_unit}\n{association}"
    return rbs_per_unit


def distribute_rbs_ues(
    rbs_per_ue: np.ndarray,
    allocation_rbs: np.ndarray,
    slice_ues: np.ndarray,
    rbs_per_slice: np.ndarray,
    slice_idx: int,
) -> np.ndarray:
    rb_idx = np.sum(rbs_per_slice[:slice_idx], dtype=int)
    for idx, ue_idx in enumerate(slice_ues):
        allocation_rbs[
            0, ue_idx, rb_idx : rb_idx + rbs_per_ue[idx].astype(int)
        ] = 1
        rb_idx += rbs_per_ue[idx].astype(int)

    return allocation_rbs


def round_int_equal_sum(
    float_array: np.ndarray, target_sum: int
) -> np.ndarray:
    non_zero_indices = np.where(float_array != 0)[0]
    non_zero_values = float_array[non_zero_indices]

    # Proportional distribution to get as close as possible to the target sum
    proportional_integers = np.floor(
        target_sum * non_zero_values / np.sum(non_zero_values)
    ).astype(int)

    # Calculate the remaining adjustment
    adjustment = target_sum - np.sum(proportional_integers)

    # Distribute the remaining adjustment among the highest values
    sorted_indices = np.argsort(non_zero_values)[::-1]
    for i in range(adjustment):
        index = sorted_indices[i % len(sorted_indices)]
        proportional_integers[index] += 1

    # Reconstruct the rounded result
    rounded_integers = np.zeros_like(float_array, dtype=int)
    rounded_integers[non_zero_indices] = proportional_integers

    return rounded_integers


def round_robin(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    last_unformatted_obs: deque,
    distribute_rbs: bool = True,
    account_buffer: bool = True,
) -> np.ndarray:
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    slice_ues_buffer = slice_ues
    if account_buffer:
        slice_ues_buffer = slice_ues[
            np.logical_not(np.isclose(buffer_occ, np.zeros_like(buffer_occ)))
        ]  # Consider only UEs that have packets available in the buffer
        if slice_ues_buffer.shape[0] == 0:
            slice_ues_buffer = slice_ues
    rbs_per_ue = np.ones_like(slice_ues_buffer, dtype=float) * np.floor(
        rbs_per_slice[slice_idx] / slice_ues_buffer.shape[0]
    )
    remaining_rbs = int(rbs_per_slice[slice_idx] % slice_ues_buffer.shape[0])
    rbs_per_ue[0:remaining_rbs] += 1
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "RR: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"

    if distribute_rbs:
        allocation_rbs = distribute_rbs_ues(
            rbs_per_ue,
            allocation_rbs,
            slice_ues_buffer,
            rbs_per_slice,
            slice_idx,
        )
        assert (
            np.sum(allocation_rbs[0, slice_ues_buffer, :])
            == rbs_per_slice[slice_idx]
        ), "Distribute RBs is different from RR distribution"
        assert np.sum(allocation_rbs) == np.sum(
            rbs_per_slice[0 : slice_idx + 1]
        ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

        return allocation_rbs
    else:
        return rbs_per_ue


def proportional_fairness(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    env: MARLCommEnv,
    last_unformatted_obs: deque,
    num_available_rbs: np.ndarray,
) -> np.ndarray:
    spectral_eff = np.mean(
        last_unformatted_obs[0]["spectral_efficiencies"][0, slice_ues, :],
        axis=1,
    )
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    throughput_available = np.minimum(
        spectral_eff
        * (
            rbs_per_slice[slice_idx]
            * env.comm_env.config.basestation_config.bandwidths[0]
            / num_available_rbs[0]
        )
        / slice_ues.shape[0],
        buffer_occ
        * env.comm_env.components.ues.max_buffer_pkts[slice_ues]
        * env.comm_env.components.ues.pkt_sizes[slice_ues],
    )
    pkt_snt_throughput = np.mean(
        [
            last_unformatted_obs[idx]["pkt_effective_thr"][slice_ues]
            for idx in range(len(last_unformatted_obs))
        ],
        axis=0,
    )
    # pkt_effective_thr 已经是 bit/step，无需再次乘 pkt_size
    snt_throughput = pkt_snt_throughput
    snt_throughput[
        np.isclose(throughput_available, np.zeros_like(throughput_available))
    ] = 1
    weights = np.divide(
        throughput_available,
        snt_throughput,
        where=np.logical_not(
            np.isclose(snt_throughput, np.zeros_like(snt_throughput))
        ),
        out=2 * np.max(throughput_available) * np.ones_like(snt_throughput),
    )
    rbs_per_ue = (
        round_int_equal_sum(
            rbs_per_slice[slice_idx] * weights / np.sum(weights),
            rbs_per_slice[slice_idx],
        )
        if np.sum(weights) != 0
        else round_robin(
            allocation_rbs=np.array([]),
            slice_idx=slice_idx,
            rbs_per_slice=rbs_per_slice,
            slice_ues=slice_ues,
            last_unformatted_obs=last_unformatted_obs,
            distribute_rbs=False,
            account_buffer=False,
        )
    )
    allocation_rbs = distribute_rbs_ues(
        rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
    )

    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "PF: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
    ), "Distribute RBs is different from RR distribution"
    assert np.sum(allocation_rbs) == np.sum(
        rbs_per_slice[0 : slice_idx + 1]
    ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

    return allocation_rbs


def max_throughput(
    allocation_rbs: np.ndarray,
    slice_idx: int,
    rbs_per_slice: np.ndarray,
    slice_ues: np.ndarray,
    env: MARLCommEnv,
    last_unformatted_obs: deque,
    num_available_rbs: np.ndarray,
) -> np.ndarray:
    spectral_eff = np.mean(
        last_unformatted_obs[0]["spectral_efficiencies"][0, slice_ues, :],
        axis=1,
    )
    buffer_occ = last_unformatted_obs[0]["buffer_occupancies"][slice_ues]
    throughput_available = np.minimum(
        spectral_eff
        * (
            rbs_per_slice[slice_idx]
            * env.comm_env.config.basestation_config.bandwidths[0]
            / num_available_rbs[0]
        )
        / slice_ues.shape[0],
        buffer_occ
        * env.comm_env.components.ues.max_buffer_pkts[slice_ues]
        * env.comm_env.components.ues.pkt_sizes[slice_ues],
    )
    rbs_per_ue = (
        round_int_equal_sum(
            rbs_per_slice[slice_idx]
            * throughput_available
            / np.sum(throughput_available),
            rbs_per_slice[slice_idx],
        )
        if np.sum(throughput_available) != 0
        else round_robin(
            allocation_rbs=np.array([]),
            slice_idx=slice_idx,
            rbs_per_slice=rbs_per_slice,
            slice_ues=slice_ues,
            last_unformatted_obs=last_unformatted_obs,
            distribute_rbs=False,
            account_buffer=False,
        )
    )
    allocation_rbs = distribute_rbs_ues(
        rbs_per_ue, allocation_rbs, slice_ues, rbs_per_slice, slice_idx
    )

    assert (
        np.sum(rbs_per_ue < 0) == 0
    ), "Negative RBs on rbs_per_ue are not allowed"
    assert (
        np.sum(rbs_per_ue) == rbs_per_slice[slice_idx]
    ), "MT: Number of allocated RBs is different than available RBs"
    assert (
        np.sum(allocation_rbs[0, slice_ues, :]) == rbs_per_slice[slice_idx]
    ), "Distribute RBs is different from RR distribution"

    assert np.sum(allocation_rbs) == np.sum(
        rbs_per_slice[0 : slice_idx + 1]
    ), f"allocation_rbs is different from rbs_per_slice at slice {slice_idx}"

    return allocation_rbs


