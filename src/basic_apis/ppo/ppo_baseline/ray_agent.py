# 标准库
import os
import re
from random import choice
from typing import Callable, List, Optional, Union
from typing import Dict as TYPE_DICT

# 第三方库
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Dict

# Ray 核心
import ray
from ray import air, tune

# Ray Tune
from ray.tune.logger import JsonLoggerCallback, CSVLoggerCallback, TBXLoggerCallback
from ray.tune.registry import register_env
from ray.tune.schedulers.async_hyperband import AsyncHyperBandScheduler
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)

# Ray RLlib - 核心组件
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import Policy, PolicySpec


# Ray RLlib - 模型相关
from ray.rllib.models import ModelCatalog
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

# Ray RLlib - 工具类
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

# 本地模块
from src.basic_apis.network_slicing_business.path_manager import PathManager


class TorchActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of ActionMaskingModel."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        TorchModelV2.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
        )
        nn.Module.__init__(self)

        if hasattr(obs_space, "original_space"):
            tmp_obs_space = obs_space.original_space
        else:
            tmp_obs_space = obs_space
        self.internal_model = TorchFC(
            tmp_obs_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

    def forward(self, input_dict, state, seq_lens):
        # --- 关键修复：确保设备和类型一致 ---

        # 1. 从内部模型获取正确的设备 (device)
        #    这是获取当前模型所在设备的最可靠方法
        device = next(self.internal_model.parameters()).device

        # 2. 提取数据，并确保它们在正确的设备和类型上
        action_mask = input_dict["obs"]["action_mask"].to(
            device=device, dtype=torch.float32
        )
        observations = input_dict["obs"]["observations"].to(
            device=device, dtype=torch.float32
        )
        # ------------------------------------

        # 排序逻辑保持不变
        action_mask_sorted = torch.zeros_like(action_mask)
        active_slices = int(torch.sum(action_mask[0]).item())
        if active_slices > 0:
            action_mask_sorted[:, -active_slices:] = 1

        # 计算 logits，它现在肯定在正确的设备和类型上
        logits, _ = self.internal_model({"obs": observations})

        # 现在，logits 和 action_mask_sorted 都在同一个设备和类型上，
        # torch.cat 操作是安全的。
        expanded_logits = torch.cat((logits, action_mask_sorted), 1)

        return expanded_logits, state

    def value_function(self):
        return self.internal_model.value_function()

class TorchDiagGaussian(TorchDistributionWrapper):
    """Wrapper class for PyTorch Normal distribution."""

    @override(ActionDistribution)
    def __init__(
        self,
        inputs: List[TensorType],
        model: TorchModelV2,
        *,
        action_space: Optional[gym.spaces.Space] = None,
    ):
        super().__init__(inputs, model)
        assert isinstance(
            self.inputs, torch.Tensor
        ), "Inputs must be a torch.Tensor"
        mean, log_std, masks = torch.chunk(self.inputs, 3, dim=1)
        log_std = torch.exp(log_std)

        # Modify values based on masks using torch.where
        log_std = torch.where(masks == 0, torch.tensor(1e-9), log_std)
        mean = torch.where(masks == 0, torch.tensor(-1), mean)
        self.dist = torch.distributions.normal.Normal(mean, log_std)
        # Remember to squeeze action samples in case action space is Box(shape)
        self.zero_action_dim = action_space and action_space.shape == ()

    @override(TorchDistributionWrapper)
    def sample(self) -> TensorType:
        sample = super().sample()
        if self.zero_action_dim:
            return torch.squeeze(sample, dim=-1)
        return sample

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        self.last_sample = self.dist.mean
        return self.last_sample

    @override(TorchDistributionWrapper)
    def logp(self, actions: TensorType) -> TensorType:
        return super().logp(actions).sum(-1)

    @override(TorchDistributionWrapper)
    def entropy(self) -> TensorType:
        return super().entropy().sum(-1)

    @override(TorchDistributionWrapper)
    def kl(self, other: ActionDistribution) -> TensorType:
        return super().kl(other).sum(-1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ) -> Union[int, np.ndarray]:
        # fake_action_space = action_space["rb_allocation"]
        # assert fake_action_space.shape is not None, "Action space shape must be set"
        # actual_shape = tuple([fake_action_space.shape[0] * 2])
        assert action_space.shape is not None, "Action space shape must be set"
        output_size = int(np.prod(action_space.shape, dtype=np.int32) * 2)

        return output_size

ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)


class RayAgent:
    # It does not implement any of the MARL logic. It is a wrapper to use Ray
    # with the general simulation script simu.py
    def __init__(
        self,
        env_creator: Callable,
        env_config: dict,
        debug_mode: bool = False,
        enable_masks: bool = True,
        restore: bool = False,
        param_config_mode: str = "default",
        param_config_scenario: Optional[str] = None,
        param_config_agent: Optional[str] = None,
        stochastic_policy: bool = False,
        hyper_opt_algo: Optional[str] = None,
        hyper_opt_enable: bool = False,
        shared_policies: bool = True,
        number_rollout_workers: int = 0,
        path_manager: PathManager = None,
    ):
        if not ray.is_initialized():
            ray.init(local_mode=debug_mode, num_cpus=128, num_gpus=4)
            print("✅ Ray initialized successfully")
        else:
            print("⚠️ Ray already initialized, skipping init")
        register_env("marl_comm_env", lambda config: env_creator(config, True))
        ModelCatalog.register_custom_action_dist(
            "masked_gaussian", TorchDiagGaussian
        )

        self.stochastic_policy = stochastic_policy
        self.restore = restore
        self.env_config = env_config
        self.agent = None
        self.enable_masks = enable_masks
        self.path_manager = path_manager
        self.read_checkpoint = path_manager.get_read_checkpoint_path()
        self.algo = None
        self.steps_per_episode = 1000
        self.number_rollout_workers = number_rollout_workers
        self.min_eps_iteration_checkpoint = 2
        self.hyper_opt_algo = hyper_opt_algo
        self.hyper_opt_enable = hyper_opt_enable
        self.shared_policies = shared_policies
        self.maximum_number_slices = 5
        self.net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
            "large": [256, 256, 256],
            "verybig": [512, 512, 512],
        }

        # Hyperparameter optimizer configuration
        if self.hyper_opt_algo == "asha":
            # 1. 增加样本数量以获得更好的搜索覆盖
            self.num_samples = 100
            self.time_attr = "timesteps_total"
            self.brackets = 1
            self.max_t = 500 * self.steps_per_episode
            self.grace_period = 80 * self.steps_per_episode
            self.reduction_factor = 2
            train_batch_size_options = np.array(
                [
                    8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                ]
            )

            # 2. 修复超参数搜索空间定义
            self.initial_hyperparam = {
                # 学习率：扩大搜索范围，包含更小的学习率
                "lr": tune.loguniform(1e-6, 5e-4),

                # 批次大小：使用独立的参数
                # "sgd_minibatch_size": tune.choice([8, 16, 32, 64, 128, 256, 512]),
                "sgd_minibatch_size": tune.choice([8, 16, 32, 64, 128, 256, 512]),

                # "train_batch_size": tune.sample_from(
                #     lambda config: np.random.choice(
                #         [
                #             config["sgd_minibatch_size"] * 2**i
                #             for i in range(
                #                 train_batch_size_options.shape[0]
                #                 - (
                #                     train_batch_size_options
                #                     == config["sgd_minibatch_size"]
                #                 ).nonzero()[0][0],
                #             )
                #         ],
                #     )
                # ),
                "train_batch_size": 1024,
                # 折扣因子：针对长期奖励优化
                "gamma": tune.choice([0.95, 0.98, 0.99, 0.995, 0.999, 0.9995]),

                # 训练轮次：增加选择范围
                "num_epochs": tune.choice([3, 5, 10, 15, 20, 30]),

                # GAE参数：更精细的搜索
                "lambda": tune.choice([0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99]),

                # PPO裁剪参数：扩大搜索范围
                "clip_param": tune.choice([0.1, 0.15, 0.2, 0.25, 0.3, 0.35]),

                # 熵系数：对探索很重要
                "entropy_coeff": tune.loguniform(1e-8, 0.2),

                # 价值函数损失系数
                "vf_loss_coeff": tune.uniform(0.1, 1.5),

                # 梯度裁剪：更细粒度的搜索
                "grad_clip": tune.choice([0.3, 0.5, 0.7, 1.0, 1.5, 2.0]),

                # 网络架构：增加更多选择
                "net_arch": tune.choice([
                    [64, 64],  # small
                    [128, 128],  # medium-small
                    [256, 256],  # medium
                    [512, 512],  # large
                    [256, 256, 256],  # deep medium
                    [512, 512, 512],  # deep large
                    [400, 300],  # asymmetric
                    [64, 128, 64],  # funnel
                    [128, 256, 128],  # larger funnel
                ]),

                # 新增的超参数
                "kl_coeff": tune.choice([0.0, 0.2, 0.5, 1.0]),
                "kl_target": tune.choice([0.01, 0.02, 0.05, 0.1]),
                "vf_clip_param": tune.choice([10.0, 50.0, 100.0, np.inf]),
                "activation_functions": tune.choice(["relu", "tanh", "elu"]),
                "initialization_methods": tune.choice(["xavier", "he", "orthogonal"])
            }
        elif self.hyper_opt_algo is None:
            self.initial_hyperparam = None
        else:
            raise ValueError(f"Invalid hyper_opt_algo: {self.hyper_opt_algo}.")

        # Initial hyperparameter loading in case not using hyperparameter optimization
        if param_config_mode == "default":
            # self.param_config = {
            #     "lr": 0.0003,
            #     "train_batch_size": 1000,
            #     "sgd_minibatch_size": 64,
            #     "num_epochs": 10,
            #     "gamma": 0.99,
            #     "lambda": 0.95,
            #     "net_arch": self.net_arch["small"],
            #     "clip_param": 0.2,
            #     "entropy_coeff": 0.01,
            #     "vf_loss_coeff": 0.5,
            #     "grad_clip": 0.5,
            # }
            self.param_config = {
                "lr": 0.001,                    # ⚠️ 学习率过大（3倍）
                "train_batch_size": 1000,
                "sgd_minibatch_size": 64,
                "num_epochs": 20,               # ⚠️ epoch过多（2倍）
                "gamma": 0.99,
                "lambda": 0.95,
                "net_arch": self.net_arch["small"],
                "clip_param": 0.4,              # ⚠️ clip范围过大（2倍）
                "entropy_coeff": 0.001,         # ⚠️ 探索系数过小（1/10）
                "vf_loss_coeff": 0.5,
                "grad_clip": 0.5,
            }
        elif param_config_mode in [
            "checkpoint",
            "checkpoint_avg",
            "checkpoint_avg_peaks",
        ]:
            self.param_config = self.load_config(
                param_config_mode, param_config_agent, param_config_scenario
            )
        elif param_config_mode == "pre_computed":
            # Computed using hyperparam_opt_mult_slice scenario
            self.param_config = {
                "lr": 6.1494053683206764e-06,
                "sgd_minibatch_size": 16,
                "train_batch_size": 1000,
                "gamma": 0.6,
                "num_epochs": 10,
                "lambda": 0.95,
                "net_arch": self.net_arch["verybig"],
                "clip_param": 0.2,
                "entropy_coeff": 0.014410343410248648,
                "vf_loss_coeff": 0.42179598812262487,
                "grad_clip": 0.5,
            }
        else:
            raise ValueError(
                f"Invalid param_config_mode: {param_config_mode}."
            )

        self.eps_per_iteration = (
            self.param_config["train_batch_size"] // self.steps_per_episode
            if self.param_config["train_batch_size"] >= self.steps_per_episode
            else self.param_config["train_batch_size"] / self.steps_per_episode
        )

    def train(self, total_timesteps: int=0):
        # Total timesteps is not used in this implementation
        # it is just a placeholder to keep the same interface as SB3

        algo_config = self.gen_config(self.env_config)
        scenario_mode = self.env_config["scenario_mode"]
        stop = {
            "timesteps_total": int(
                (
                        self.env_config[scenario_mode]['training']["max_episode"]
                        - self.env_config[scenario_mode]['training']["initial_episode"]
                )
                * self.env_config["training_epochs"]
                * self.steps_per_episode
                * len(self.env_config[scenario_mode]['training']['active_scenario_list'])
            ),
        }

        # Whether to use a hyperparameter opt algo
        if self.hyper_opt_enable:
            if self.hyper_opt_algo == "asha":
                asha = AsyncHyperBandScheduler(
                    time_attr=self.time_attr,
                    grace_period=self.grace_period,
                    max_t=self.max_t,
                    reduction_factor=self.reduction_factor,
                    brackets=self.brackets,
                    stop_last_trials=True,
                )
                tune_config = tune.TuneConfig(
                    metric="evaluation/env_runners/episode_return_mean",
                    mode="max",
                    scheduler=asha,
                    num_samples=self.num_samples,
                    max_concurrent_trials=4
                )
            elif self.hyper_opt_algo is None:
                tune_config = None
            else:
                raise ValueError(
                    f"Invalid hyper_opt_algo: {self.hyper_opt_algo}."
                )
        else:
            tune_config = None

        tuner = tune.Tuner(
            "PPO",
            param_space=algo_config.to_dict(),
            tune_config=tune_config,
            run_config=air.RunConfig(
                storage_path=f"{self.read_checkpoint}/",
                name=self.env_config["agent"],
                stop=stop,
                verbose=2,
                checkpoint_config=air.CheckpointConfig(
                    checkpoint_frequency=np.rint(
                        self.env_config["checkpoint_episode_freq"]
                        / self.eps_per_iteration
                    ).astype(int),
                    checkpoint_at_end=True,
                ),
                callbacks=[
                    JsonLoggerCallback(),
                    CSVLoggerCallback(),
                    TBXLoggerCallback()
                ] if not self.hyper_opt_enable else None
            ),
        )
        results = tuner.fit()
        print(results)

    def gen_config(self, env_config):
        if not self.hyper_opt_enable or self.initial_hyperparam is None:
            params = self.param_config
        else:
            params = self.initial_hyperparam

        policies = self.generate_policies(self.shared_policies, self.maximum_number_slices)
        print(f"🔍 Debug: Generated policies: {list(policies.keys())}")

        algo_config = (
            PPOConfig()
            .environment(env="marl_comm_env", env_config=env_config)
            .framework("torch")
            .multi_agent(
                policies=self.generate_policies(
                    self.shared_policies, self.maximum_number_slices
                ),
                policy_mapping_fn=(
                    self.policy_mapping_fn_shared
                    if self.shared_policies
                    else self.policy_mapping_fn_non_shared
                ),
            )
            .env_runners(
                num_env_runners=self.number_rollout_workers,
                num_envs_per_env_runner=1,
                num_cpus_per_env_runner=2,
                num_gpus_per_env_runner=0,  # 通常为 0
            )
            .training(
                gamma=params["gamma"],
                lr=params["lr"],
                lambda_=params["lambda"],
                num_epochs=params["num_epochs"],
                clip_param=params["clip_param"],
                entropy_coeff=params["entropy_coeff"],
                vf_loss_coeff=params["vf_loss_coeff"],
                grad_clip=params["grad_clip"],
                # 这个 model 配置将作为未使用 custom_model 的策略的默认配置
                model={"fcnet_hiddens": params["net_arch"]},
                vf_clip_param=np.inf,
                use_gae=True,
                kl_coeff=0,
                use_kl_loss=False,
                kl_target=0,
            )
            .debugging(seed=env_config["seed"])
            .reporting(metrics_num_episodes_for_smoothing=1)
            .callbacks(UpdatePolicyCallback)
            .resources(
                num_cpus_for_main_process=4,  # 建议为 >1

                # 为主训练器（Learner）分配的 GPU 数量
                num_gpus=0.25 if torch.cuda.is_available() else 0,
            )
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            # ----------------------------------------------------------------------
        )

        # 顶层属性赋值
        algo_config.train_batch_size = params["train_batch_size"]
        algo_config.sgd_minibatch_size = params["sgd_minibatch_size"]
        if hasattr(algo_config, 'minibatch_size'):
            algo_config.minibatch_size = params["sgd_minibatch_size"]

        scenario_mode = env_config["scenario_mode"]
        if self.env_config["enable_evaluation"]:
            algo_config.evaluation(
                # evaluation_interval=np.rint(
                #     env_config["episode_evaluation_freq"] / self.eps_per_iteration
                # ).astype(int),
                evaluation_interval=1,
                evaluation_duration=env_config[scenario_mode]['evaluating']["evaluating_episodes"],
                evaluation_duration_unit="episodes",
                evaluation_num_env_runners=2,
                evaluation_parallel_to_training=True,
                evaluation_config={
                    "explore": self.stochastic_policy,
                    "env_config": dict(
                        env_config,
                        initial_episode_number=env_config[scenario_mode]['evaluating']["initial_episode"],
                        max_episode_number=(
                                env_config[scenario_mode]['evaluating']["initial_episode"]
                                + env_config[scenario_mode]['evaluating']["evaluating_episodes"]
                        ),
                    ),
                },
            )

        print(f"🔍 Debug: Multi-agent config:")
        print(f"  - Shared policies: {self.shared_policies}")
        print(f"  - Policy mapping function: {'shared' if self.shared_policies else 'non_shared'}")

        return algo_config

    def action_mask_policy(self):
        config = PPOConfig.overrides(
            model={
                "custom_model": TorchActionMaskModel,
                "custom_action_dist": "masked_gaussian",
            },
        )
        return PolicySpec(config=config)

    def generate_policies(
            self, shared_policies: bool, max_number_slices: int
    ) -> TYPE_DICT[str, PolicySpec]:
        # 定义使用您的自定义模型的配置
        action_mask_model_config = {
            "custom_model": "torch_action_mask_model",  # 使用您注册的字符串名称
            "custom_action_dist": "masked_gaussian",
        }

        if shared_policies:
            return {
                "inter_slice_sched": PolicySpec(
                    # 如果启用掩码，则应用此配置
                    config={"model": action_mask_model_config} if self.enable_masks else {}
                ),
                "intra_slice_sched": PolicySpec(),  # 这个策略使用默认模型
            }
        else:
            # 非共享策略的逻辑... (如果需要，也可以在这里应用)
            policies = {
                "inter_slice_sched": PolicySpec(
                    config={"model": action_mask_model_config} if self.enable_masks else {}
                ),
            }
            for i in range(max_number_slices):
                policies[f"intra_slice_sched_{i}"] = PolicySpec()
            return policies

    def policy_mapping_fn_shared(
            self, agent_id, episode=None, worker=None, **kwargs
    ):
        agent_idx = int(agent_id.partition("_")[2])

        return "inter_slice_sched" if agent_idx == 0 else "intra_slice_sched"

    def policy_mapping_fn_non_shared(
            self, agent_id, episode=None, worker=None, **kwargs
    ):
        agent_idx = int(agent_id.partition("_")[2])

        return (
            "inter_slice_sched"
            if agent_idx == 0
            else f"intra_slice_sched_{agent_idx - 1}"
        )

    def load(
            self, agent_name, scenario, method="last", finetune=False
    ) -> None:
        if not finetune:
            if method == "last":
                analysis = tune.ExperimentAnalysis(
                    f"{self.read_checkpoint}/{scenario}/{agent_name}/"
                )
                assert analysis.trials is not None, "Analysis trial is None"
                checkpoint = analysis.get_last_checkpoint(analysis.trials[0])
            elif method == "best":
                analysis = tune.ExperimentAnalysis(
                    f"{self.read_checkpoint}/{scenario}/{agent_name}/"
                )
                assert analysis.trials is not None, "Analysis trial is None"
                checkpoint = analysis.get_best_checkpoint(
                    analysis.trials[0],
                    "evaluation/env_runners/policy_reward_mean/inter_slice_sched",
                    "max",
                )
            elif method == "best_train":
                analysis = tune.ExperimentAnalysis(
                    f"{self.read_checkpoint}/{scenario}/{agent_name}/"
                )
                assert analysis.trials is not None, "Analysis trial is None"
                checkpoint = analysis.get_best_checkpoint(
                    analysis.trials[0],
                    "env_runners/policy_reward_mean/inter_slice_sched",
                    "max",
                )
            elif method[0:6] == "trial_":
                directory = f"{self.read_checkpoint}/{scenario}/{agent_name}/"
                trial_number = method.split("_")[1]
                checkpoint_number = method.split("_")[3]
                folder_pattern = re.compile(rf".*{re.escape(trial_number)}.*")
                folders = os.listdir(directory)
                matching_folders = [
                    folder
                    for folder in folders
                    if os.path.isdir(os.path.join(directory, folder))
                       and folder_pattern.match(folder)
                ]
                checkpoint_pattern = re.compile(
                    rf".*{re.escape(checkpoint_number)}.*"
                )
                checkpoint_folders = os.listdir(
                    directory + matching_folders[0]
                )
                checkpoint_matching_folders = [
                    folder
                    for folder in checkpoint_folders
                    if os.path.isdir(
                        os.path.join(directory + matching_folders[0], folder)
                    )
                       and checkpoint_pattern.match(folder)
                ]
                checkpoint = (
                        directory
                        + matching_folders[0]
                        + "/"
                        + checkpoint_matching_folders[0]
                )
            elif isinstance(method, int):  # TODO check if correct
                raise NotImplementedError(
                    "Checkpoint by iteration not implemented"
                )
            else:
                raise ValueError(f"Invalid method {method} for finetune load")
            assert checkpoint is not None, "Ray checkpoint is None"
            self.algo = Algorithm.from_checkpoint(checkpoint)

    def load_config(self, mode, agent_name, scenario) -> dict:
        metric = "evaluation/env_runners/policy_reward_mean/inter_slice_sched"
        assert isinstance(
            self.initial_hyperparam, dict
        ), "Initial hyperparam is not a dictionary"
        hyperparameters = list(self.initial_hyperparam.keys())
        hyperparameters.remove("net_arch")
        analysis = tune.ExperimentAnalysis(
            f"{self.read_checkpoint}/{scenario}/{agent_name}/"
        )
        assert analysis.trials is not None, "Analysis trial is None"
        if mode == "checkpoint":
            config = analysis.get_best_config(metric=metric, mode="max")
        elif mode == "checkpoint_avg":
            trial_dfs = analysis.trial_dataframes
            trials_avg = {}
            for trial_name in trial_dfs.keys():
                if metric in trial_dfs[trial_name].columns:
                    trial_df = (
                        trial_dfs[trial_name][metric].dropna().to_numpy()
                    )
                    if trial_df.shape[0] >= 10:
                        trials_avg[trial_name] = np.mean(trial_df)
            best_trial_name = max(trials_avg, key=lambda key: trials_avg[key])
            config = analysis.get_all_configs()[best_trial_name]
        elif mode == "checkpoint_avg_peaks":
            peaks_number = 10
            trial_dfs = analysis.trial_dataframes
            trials_avg = {}
            for trial_name in trial_dfs.keys():
                if metric in trial_dfs[trial_name].columns:
                    trial_df = (
                        trial_dfs[trial_name][metric].dropna().to_numpy()
                    )
                    if trial_df.shape[0] >= peaks_number:
                        trial_df = np.sort(np.unique(trial_df))[-peaks_number:]
                        trials_avg[trial_name] = np.mean(trial_df)
            best_trial_name = max(trials_avg, key=lambda key: trials_avg[key])
            config = analysis.get_all_configs()[best_trial_name]
        else:
            raise ValueError(f"Invalid mode {mode} for load_config")
        assert isinstance(config, dict), "Config is not a dictionary"
        selected_config = {key: config[key] for key in hyperparameters}
        selected_config["net_arch"] = config["model"]["fcnet_hiddens"]

        return selected_config

    def step(self, obs):
        action = {}
        assert isinstance(obs, dict), "Observations must be a dictionary."
        assert isinstance(
            self.algo, Algorithm
        ), "Algorithm must be an instance of Algorithm."
        for agent_id, agent_obs in obs.items():
            policy_id = (
                self.policy_mapping_fn_shared(agent_id)
                if self.shared_policies
                else self.policy_mapping_fn_non_shared(agent_id)
            )
            print(f"🔍 Debug: Agent {agent_id} -> Policy {policy_id}")
            action[agent_id] = self.algo.compute_single_action(
                agent_obs,
                policy_id=policy_id,
                explore=self.stochastic_policy,
            )
        return action

    def shutdown(self):
        """
        Properly shutdown the RayAgent and release all allocated resources.

        This method should be called when you're done using the RayAgent to ensure
        proper cleanup of Ray resources, GPU memory, and other allocated resources.
        """
        try:
            # 1. 清理算法实例
            if hasattr(self, 'algo') and self.algo is not None:
                try:
                    # 如果算法有cleanup方法，调用它
                    if hasattr(self.algo, 'cleanup'):
                        self.algo.cleanup()
                    # 停止算法
                    if hasattr(self.algo, 'stop'):
                        self.algo.stop()
                except Exception as e:
                    print(f"Error cleaning up algorithm: {e}")
                finally:
                    self.algo = None

            # 2. 清理任何正在运行的tune实验
            # 注意：tune.Tuner 实例通常在train()方法中是局部变量
            # 但如果有全局的tune作业在运行，可能需要手动停止

            # 3. 清除GPU缓存（如果使用了PyTorch和CUDA）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 4. 关闭Ray集群
            if ray.is_initialized():
                ray.shutdown()
                print("Ray cluster shutdown successfully.")

            print("RayAgent shutdown completed.")

        except Exception as e:
            print(f"Error during RayAgent shutdown: {e}")
            # 即使出错也要尝试关闭Ray
            if ray.is_initialized():
                ray.shutdown()

    @staticmethod
    def explore(config):
        # ensure we collect enough timesteps to do sgd
        if config["train_batch_size"] < config["sgd_minibatch_size"]:
            config["train_batch_size"] = config["sgd_minibatch_size"]
        # ensure we run at least one sgd iter
        if config["num_epochs"] < 1:
            config["num_epochs"] = 1
        return config

class UpdatePolicyCallback(DefaultCallbacks):
    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        """
        微调回调：安全地注入权重，避开 Ray 内部的 workers 属性访问 bug。
        """
        # 1. 获取路径
        finetune_path = algorithm.config.env_config.get("finetune_checkpoint_path")

        # 如果没有路径，直接返回
        if not finetune_path:
            print("ℹ️ [Fine-tuning] 未配置微调路径，从 Scratch 开始训练。")
            return

        if not os.path.exists(finetune_path):
            print(f"❌ [Fine-tuning] 路径不存在: {finetune_path}")
            return

        print(f"🔄 [Fine-tuning] 正在从 {finetune_path} 读取策略...")
        policies_dir = os.path.join(finetune_path, "policies")

        if not os.path.exists(policies_dir):
            print(f"❌ [Fine-tuning] Checkpoint 损坏，找不到 policies 目录: {policies_dir}")
            return

        # 2. 遍历硬盘上的策略文件
        # 我们不再询问 algorithm "你有几个 worker"，而是直接看 "硬盘上有几个策略"
        weights_to_set = {}
        loaded_count = 0

        policy_ids = os.listdir(policies_dir)

        for policy_id in policy_ids:
            policy_path = os.path.join(policies_dir, policy_id)

            # 确保是目录且包含权重文件
            if os.path.isdir(policy_path):
                try:
                    # 使用 Policy.from_checkpoint 加载 (纯文件操作，不涉及 Ray Cluster)
                    restored_policy = Policy.from_checkpoint(policy_path)
                    weights = restored_policy.get_weights()
                    weights_to_set[policy_id] = weights

                    print(f"   -> 已读取策略: {policy_id}")
                    loaded_count += 1

                    # 及时释放内存
                    del restored_policy
                except Exception as e:
                    print(f"⚠️ [Fine-tuning] 读取策略 {policy_id} 失败: {e}")

        # 3. 注入权重 (关键步骤)
        if weights_to_set:
            try:
                # 使用最顶层的 API，它会自动处理 workers 的同步
                # 如果某个策略在当前 agent 中不存在，RLlib 通常会忽略它或仅报警告，不会崩
                algorithm.set_weights(weights_to_set)
                print(f"🚀 [Fine-tuning] 成功注入 {loaded_count} 个策略的权重！")

            except Exception as e:
                print(f"❌ [Fine-tuning] set_weights 注入失败: {e}")

                # 最后的救命稻草：尝试使用 env_runner_group (新版 API)
                try:
                    print("🔄 [Fine-tuning] 尝试使用 env_runner_group 备选方案...")
                    if hasattr(algorithm, "env_runner_group"):
                        # 同步到本地
                        algorithm.env_runner_group.local_env_runner.set_weights(weights_to_set)
                        # 同步到远程
                        algorithm.env_runner_group.sync_weights()
                        print("🚀 [Fine-tuning] 通过 EnvRunnerGroup 注入成功！")
                except Exception as e2:
                    print(f"❌ [Fine-tuning] 备选方案也失败了: {e2}")
        else:
            print("⚠️ [Fine-tuning] 没有在 checkpoint 中找到可用的策略文件。")

    def on_evaluate_end(self, *, algorithm, evaluation_metrics, **kwargs):
        print("Evaluation metrics:", evaluation_metrics)
        episode_return_mean = evaluation_metrics.get("evaluation", {}).get("episode_return_mean")
        if episode_return_mean is not None:
            tune.report({"evaluation/env_runners/episode_return_mean": episode_return_mean})
        else:
            print("Warning: evaluation/env_runners/episode_return_mean not found")


