import numpy as np
import os
from tqdm import tqdm
import json
from typing import Tuple
from collections import deque
from src.basic_apis.ppo.utils import (
    calculate_slice_ue_obs,
    intent_drift_calc,
)


def _infer_total_steps(data_metrics) -> int:
    """Infer episode length directly from metric arrays."""
    if "slice_ue_assoc" in data_metrics:
        return int(data_metrics["slice_ue_assoc"].shape[0])
    if "pkt_effective_thr" in data_metrics:
        return int(data_metrics["pkt_effective_thr"].shape[0])
    raise ValueError("Unable to infer total steps from data_metrics")


def calculate(cfg):
    all_metric_data = {}
    data_path = {}
    for baseline in cfg.baselines:
        data_path[baseline] = []

    mode = 'model_name' # model_name or scenario_list
    if cfg.model_names and not cfg.test_scenario_list:
        mode = 'model_name'
    elif not cfg.model_names and cfg.test_scenario_list:
        mode = 'scenario_list'
    else:
        raise ValueError("Invalid scenario ")

    if mode == 'scenario_list':
        for scenario_idx in cfg.test_scenario_list:
            episode_begin = cfg.test_episode_begin + cfg.cross_scenario_skip_episode * scenario_idx
            episode_end = cfg.test_episode_end + cfg.cross_scenario_skip_episode * scenario_idx
            for episode_idx in range(episode_begin, episode_end):
                for baseline in cfg.baselines:
                    data_path[baseline].append(
                        os.path.join(cfg.data_base_dir, baseline, "metric_raw", f"scenario_{scenario_idx}", f"ep_{episode_idx}.npz")
                    )

    elif mode == 'model_name':
        for model_name in cfg.model_names:
            episode_begin, episode_end =960, 980
            for episode_idx in range(episode_begin, episode_end):
                for baseline in cfg.baselines:
                    data_path[baseline].append(
                        os.path.join(cfg.data_base_dir, baseline, "metric_raw", model_name,
                                     f"ep_{episode_idx}.npz")
                    )
    try:
        if mode == 'model_name':

            for model_name in cfg.model_names:
                for metric in cfg.metrics:
                    metric_dir = os.path.join(cfg.json_save_base_dir)
                    all_metric_data[metric] = {}

                    for baseline in cfg.baselines:
                        all_metric_data[metric][baseline] = {}
                        data_path_list = data_path.get(baseline, None)
                        if data_path_list is None:
                            raise ValueError(f"Unknown baseline: {baseline}")
                        all_metric_data[metric][baseline]["nhp"], all_metric_data[metric][baseline][
                            "hp"] = get_metric_episodes(metric, data_path_list)
                        baseline_path = os.path.join(metric_dir, baseline, "metric_json", model_name, f"{metric}.json")
                        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
                        json_str = json.dumps(
                            all_metric_data[metric][baseline],
                            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o.item() if isinstance(o, np.generic) else o,
                            indent=4
                        )
                        with open(baseline_path, 'w', encoding='utf-8') as f:
                            f.write(json_str)

        elif mode == 'scenario_list':
            for scenario_idx in cfg.test_scenario_list:
                for metric in cfg.metrics:
                    metric_dir = os.path.join(cfg.json_save_base_dir)
                    all_metric_data[metric] = {}

                    for baseline in cfg.baselines:
                        all_metric_data[metric][baseline] = {}
                        data_path_list = data_path.get(baseline, None)
                        if data_path_list is None:
                            raise ValueError(f"Unknown baseline: {baseline}")
                        all_metric_data[metric][baseline]["nhp"], all_metric_data[metric][baseline][
                            "hp"] = get_metric_episodes(metric, data_path_list)
                        baseline_path = os.path.join(metric_dir, baseline, "metric_json", f"scenario_{scenario_idx}", f"{metric}.json")
                        os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
                        json_str = json.dumps(
                            all_metric_data[metric][baseline],
                            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o.item() if isinstance(o,
                                                                                                                  np.generic) else o,
                            indent=4
                        )
                        with open(baseline_path, 'w', encoding='utf-8') as f:
                            f.write(json_str)

    except Exception as e:
        tqdm.write(f"Error in fetching metric data: {e}")
        import traceback
        traceback.print_exc()



def get_metric_episodes(
    metric, metric_data_path_list
) -> Tuple[np.ndarray, np.ndarray]:
    y_values = np.array([])
    y2_values = np.array([])
    # for path in metric_data_path_list:
    for path in tqdm(metric_data_path_list, desc=f"Generating metric {metric} data..."):
        data = np.load(
            # f"hist/{scenario}/{agent}/ep_{episode}.npz",
            path,
            allow_pickle=True,
        )
        data_metrics = {
            "pkt_incoming": data["pkt_incoming"],
            "pkt_throughputs": data["pkt_throughputs"],
            "pkt_effective_thr": data["pkt_effective_thr"],
            "buffer_occupancies": data["buffer_occupancies"],
            "buffer_latencies": data["buffer_latencies"],
            "dropped_pkts": data["dropped_pkts"],
            "mobility": data["mobility"],
            "spectral_efficiencies": data["spectral_efficiencies"],
            "basestation_ue_assoc": data["basestation_ue_assoc"],
            "basestation_slice_assoc": data["basestation_slice_assoc"],
            "slice_ue_assoc": data["slice_ue_assoc"],
            "sched_decision": data["sched_decision"],
            # "reward": data["reward"],
            "slice_req": data["slice_req"],
            # "obs": data["obs"],
            # "agent_action": data["agent_action"],
        }
        match metric:
            # case "reward_per_episode" | "reward_per_episode_cumsum":
            #     reward = (
            #         [
            #             data_metrics["reward"][idx]["player_0"]
            #             for idx in range(data_metrics["reward"].shape[0])
            #         ]
            #         # if "ib_sched" in agent
            #         # else data_metrics["reward"]
            #     )
            #     y_values = np.append(y_values, np.sum(reward))
            #     y2_values = np.array([])
            case "violations_per_episode" | "violations_per_episode_cumsum":
                violations, _, _, _ = calc_slice_violations(data_metrics)
                violations_pri, _, _, _ = calc_slice_violations(
                    data_metrics, priority=True
                )
                y_values = np.append(y_values, np.sum(violations))
                y2_values = np.append(y2_values, np.sum(violations_pri))
            case (
                "normalized_violations_per_episode"
                | "normalized_violations_per_episode_cumsum"
            ):
                violations, _, _, _ = calc_slice_violations(data_metrics)
                active_slices_episode = (
                    np.sum(data_metrics["basestation_slice_assoc"][0])
                    * violations.shape[0]
                )
                y_values = np.append(
                    y_values, np.sum(violations) / active_slices_episode
                )
                active_slices_episode_pri = (
                    np.sum(
                        [
                            data_metrics["slice_req"][0][slice]["priority"]
                            for slice in data_metrics["slice_req"][0]
                            if data_metrics["slice_req"][0][slice] != {}
                        ]
                    )
                    * violations.shape[0]
                )
                violations_pri, _, _, _ = calc_slice_violations(
                    data_metrics, priority=True
                )
                y2_values = np.append(
                    y2_values,
                    (
                        np.sum(violations_pri) / active_slices_episode_pri
                        if active_slices_episode_pri > 0
                        else 0
                    ),
                )
            case "distance_fulfill" | "distance_fulfill_cumsum":
                distance = calc_intent_distance(data_metrics)
                y_values = np.append(y_values, np.sum(distance))
                distance_pri = calc_intent_distance(
                    data_metrics, priority=True
                )
                y2_values = np.append(y2_values, np.sum(distance_pri))
            case (
                "normalized_distance_fulfill"
                | "normalized_distance_fulfill_cumsum"
            ):
                distance = calc_intent_distance(data_metrics)
                active_slices_episode = (
                    np.sum(data_metrics["basestation_slice_assoc"][0])
                    * distance.shape[0]
                )
                y_values = np.append(
                    y_values, np.sum(distance) / active_slices_episode
                )
                active_slices_episode_pri = (
                    np.sum(
                        [
                            data_metrics["slice_req"][0][slice]["priority"]
                            for slice in data_metrics["slice_req"][0]
                            if data_metrics["slice_req"][0][slice] != {}
                        ]
                    )
                    * distance.shape[0]
                )
                distance_pri = calc_intent_distance(
                    data_metrics, priority=True
                )
                y2_values = np.append(
                    y2_values,
                    (
                        np.sum(distance_pri) / active_slices_episode_pri
                        if active_slices_episode_pri > 0
                        else 0
                    ),
                )
    return (y_values, y2_values)

def calc_slice_violations(
    data_metrics, priority=False, slice_per_metric=False, max_number_ues_slice=5
) -> Tuple[np.ndarray, dict, np.ndarray, dict]:
    intent_drift = get_intent_drift(data_metrics)
    total_steps = _infer_total_steps(data_metrics)
    violations = np.zeros(total_steps)
    violations_per_slice_type = {}
    number_intent_metrics = 3
    metric_idxs = {
        "throughput": 0,
        "reliability": 1,
        "latency": 2,
    }
    intent_slice_metric = -2 * np.ones(
        (
            total_steps,
            data_metrics["slice_ue_assoc"][0].shape[0],
            number_intent_metrics,
        )
    )
    violations_slice_metric = {}
    for step_idx in np.arange(total_steps):
        for slice_idx in range(
                0, data_metrics["slice_ue_assoc"][step_idx].shape[0]
        ):
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][
                slice_idx
            ].nonzero()[0]
            if (
                    data_metrics["basestation_slice_assoc"][step_idx][0, slice_idx]
                    == 0
            ):
                continue
            if (
                    priority
                    and data_metrics["slice_req"][step_idx][f"slice_{slice_idx}"][
                "priority"
            ]
                    == 0
            ):
                continue
            (
                _,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                max_number_ues_slice,
                intent_drift[step_idx],
                slice_idx,
                slice_ues,
                data_metrics["slice_req"][step_idx],
            )
            intent_slice_metric[step_idx, slice_idx, :] = intent_drift_slice
            intent_drift_slice[intent_drift_slice == -2] = 1

            if slice_per_metric and np.sum(
                    intent_drift_slice < 0
            ):  # Accounts slice violation per metric
                for metric_idx in metric_idxs.keys():
                    slice_name = data_metrics["slice_req"][step_idx][
                        f"slice_{slice_idx}"
                    ]["name"]
                    if intent_drift_slice[metric_idxs[metric_idx]] < 0:
                        if slice_name in violations_slice_metric.keys():
                            if (
                                    metric_idx
                                    in violations_slice_metric[slice_name].keys()
                            ):
                                violations_slice_metric[slice_name][
                                    metric_idx
                                ] += 1
                            else:
                                violations_slice_metric[slice_name][
                                    metric_idx
                                ] = 1
                        else:
                            violations_slice_metric[slice_name] = {}
                            violations_slice_metric[slice_name][metric_idx] = 1

            intent_drift_slice = np.min(intent_drift_slice)
            slice_violation = int(
                intent_drift_slice < 0
                and not np.isclose(intent_drift_slice, -2)
            )
            violations[step_idx] += slice_violation
            if bool(slice_violation):
                slice_name = data_metrics["slice_req"][step_idx][
                    f"slice_{slice_idx}"
                ]["name"]
                if slice_name in violations_per_slice_type.keys():
                    violations_per_slice_type[slice_name] += 1
                else:
                    violations_per_slice_type[slice_name] = 1
    return (
        violations,
        violations_per_slice_type,
        intent_slice_metric,
        violations_slice_metric,
    )

def get_intent_drift(data_metrics, max_number_ues_slice=5, intent_overfulfillment_rate=0.2) -> np.ndarray:
    last_unformatted_obs = deque(maxlen=10)
    number_slices = data_metrics["slice_ue_assoc"].shape[1]
    number_ues_slice = int(
        data_metrics["slice_ue_assoc"].shape[2] / number_slices
    )
    total_steps = _infer_total_steps(data_metrics)
    intent_drift = np.zeros(
        (total_steps, number_slices, number_ues_slice, 3)
    )
    for step_idx in np.arange(total_steps):
        dict_info = {
            "pkt_effective_thr": data_metrics["pkt_effective_thr"][step_idx],
            "slice_req": data_metrics["slice_req"][step_idx],
            "buffer_occupancies": data_metrics["buffer_occupancies"][step_idx],
            "buffer_latencies": data_metrics["buffer_latencies"][step_idx],
            "slice_ue_assoc": data_metrics["slice_ue_assoc"][step_idx],
            "dropped_pkts": data_metrics["dropped_pkts"][step_idx],
        }
        last_unformatted_obs.appendleft(dict_info)
        intent_drift[step_idx, :, :, :] = intent_drift_calc(
            last_unformatted_obs,
            max_number_ues_slice,
            intent_overfulfillment_rate,
            True,
        )

    return intent_drift

def calc_intent_distance(data_metrics, priority=False, max_number_ues_slice=5) -> np.ndarray:
    intent_drift = get_intent_drift(data_metrics)
    total_steps = _infer_total_steps(data_metrics)
    distance_slice = np.zeros(total_steps)
    for step_idx in np.arange(total_steps):
        intent_array = np.array([])
        for slice_idx in range(
                0, data_metrics["slice_ue_assoc"][step_idx].shape[0]
        ):
            if (
                    data_metrics["basestation_slice_assoc"][step_idx][0, slice_idx]
                    == 0
            ):
                continue
            if (
                    priority
                    and data_metrics["slice_req"][step_idx][f"slice_{slice_idx}"][
                "priority"
            ]
                    == 0
            ):
                continue
            slice_ues = data_metrics["slice_ue_assoc"][step_idx][
                slice_idx
            ].nonzero()[0]
            (
                _,
                intent_drift_slice,
            ) = calculate_slice_ue_obs(
                max_number_ues_slice,
                intent_drift[step_idx],
                slice_idx,
                slice_ues,
                data_metrics["slice_req"][step_idx],
            )
            intent_drift_slice = np.delete(
                intent_drift_slice,
                np.logical_or(
                    np.isclose(intent_drift_slice, -2), intent_drift_slice >= 0
                ),
            )
            min_intent = (
                np.min(intent_drift_slice)
                if intent_drift_slice.shape[0] > 0
                else 0
            )
            intent_array = np.append(intent_array, min_intent)
        distance_slice[step_idx] += (
            np.sum(intent_array) if intent_array.shape[0] > 0 else 0
        )
    return distance_slice


