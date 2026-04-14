from .base_simu.simulation_factory import SimulationFactory
from .base_simu.base_simulation import BaseSimulation
from hydra.core.hydra_config import HydraConfig

from src.basic_apis.network_slicing_business.path_manager import PathManager
from src.basic_apis.asset_utils import build_versioned_run_dir, update_latest_symlink, ensure_clean_dir, ensure_dir


@SimulationFactory.register("channel_generality")
class ChannelGeneralitySimulator(BaseSimulation):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 将 path_manager 作为实例变量
        hydra_run_output_dir = HydraConfig.get().runtime.output_dir
        self.path_manager = PathManager(hydra_run_output_dir)

    def run(self):
        print("Current Simulation: Channel Generality")
        print(f"Current mode: {self.cfg.mode}")
        print("=" * 60 + f"\n")

        match self.cfg.mode:
            case "train_ppo_baseline":
                self.train_ppo_baseline()

            case "train_ppo_lstm":
                self.train_lstm_ppo()

            case "test_ppo_baseline":
                self.test_ppo_baseline()

            case "test_ppo_lstm":
                self.test_ppo_lstm()

            case "calculate_metric_value":
                self.calculate_metric_value()

            case "plot":
                self.plot()

            case "dt_dataset_collect":
                self.dt_collect_dataset()

            case "dt_dataset_collect_all":
                self.dt_collect_dataset_all()

            case "dt_dataset_collect_baseline":
                self.dt_collect_dataset_baseline()

            case "train_ppo_lstm_across_scenario":
                self.train_ppo_lstm_across_scenario()

            case "train_ppo_baseline_across_scenario":
                self.train_ppo_baseline_across_scenario()

            case "train_ppo_ha":
                self.train_ppo_ha()

            case "train_ppo_ha_weighted":
                self.train_ppo_ha_weighted()

            # case "dt_dataset_merge":
            #     self.dt_dataset_merge()
            #
            case "dt_training":
                self.dt_training()

            case "dt_testing":
                self.dt_testing()

            case "train_dt_baseline":
                self.train_dt_baseline()

            case "test_dt_baseline":
                self.test_dt_baseline()

            case "train_ppo_oneshot":
                self.train_ppo_oneshot()

            case "test_ppo_oneshot":
                self.test_ppo_oneshot()

            case "test_ppo_ha":
                self.test_ppo_ha()

            case "test_ppo_ha_weighted":
                self.test_ppo_ha_weighted()

            case "test_ppo_baseline_finetune":
                self.test_ppo_baseline_finetune()

            case "test_ppo_baseline_multi":
                self.test_ppo_baseline_multi()

            case "plot_hist":
                self.plot_hist()

            case "plot_step":
                self.plot_step()

            case "plot_finetune_entropy":
                self.plot_finetune_entropy()

            # === 新增 Baseline 模式（M5 CQL, M6 Lagrangian PPO）===
            case "cql_convert_dataset":
                self.cql_convert_dataset()

            case "train_cql":
                self.train_cql()

            case "test_cql":
                self.test_cql()

            case "train_ppo_lagrangian":
                self.train_ppo_lagrangian()

            case "test_ppo_lagrangian":
                self.test_ppo_lagrangian()

            case "train_ppo_lagrangian_baseline":
                self.train_ppo_lagrangian_baseline()

            case "test_ppo_lagrangian_baseline":
                self.test_ppo_lagrangian_baseline()

            # === 新增绘图模式（M12）===
            case "plot_transfer_curve":
                self.plot_transfer_curve()

            case "plot_sensitivity":
                self.plot_sensitivity()

            # === 推理时间基准测试（M13）===
            case "inference_benchmark":
                self.inference_benchmark()

            # === 综合结果表生成（M14）===
            case "generate_results_table":
                self.generate_results_table()

            # === 多 seed 对比绘图 ===
            case "plot_multiseed":
                self.plot_multiseed()

            case "plot_step_multiseed":
                self.plot_step_multiseed()

            case "utils":
                self.utils()

            # === dt_v2 系列（PPO-v2 teacher + DT-v2）===
            case "train_ppo_v2":
                self.train_ppo_v2()

            case "test_ppo_v2":
                self.test_ppo_v2()

            case "collect_data_v2":
                self.collect_data_v2()

            case "train_dt_v2":
                self.train_dt_v2()

            case "test_dt_v2":
                self.test_dt_v2()

            case "train_dt_v2_tiny":
                self.train_dt_v2_tiny()

            case "test_dt_v2_tiny":
                self.test_dt_v2_tiny()

            # === CQL-v3：多头离散 Q 网络（直接学习 MultiDiscrete 动作）===
            case "train_cql_v3":
                self.train_cql_v3()

            case "test_cql_v3":
                self.test_cql_v3()

            case _:
                raise ValueError(f"未知的模式: {self.cfg.mode}")

    def train_ppo_baseline(self):
        """训练基线 PPO"""
        from src.basic_apis.ppo.ppo_baseline.train_ppo_baseline import train_ppo
        train_ppo(self.cfg.train_ppo_baseline, self.path_manager)

    def train_lstm_ppo(self):
        """训练 LSTM PPO"""
        from src.basic_apis.ppo.ppo_lstm.train_recurrent_ppo import train as train_lstm_sb3
        train_lstm_sb3(self.cfg.train_ppo_lstm, path_manager=self.path_manager)

    def test_ppo_baseline(self):
        from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline
        test_ppo_baseline(self.cfg.test_ppo_baseline, self.path_manager)

    def test_ppo_lstm(self):
        """测试 LSTM PPO"""
        from src.basic_apis.ppo.ppo_lstm.test_lstm import test_ppo_lstm as test
        test(self.cfg.test_ppo_lstm,
             self.path_manager)

    def calculate_metric_value(self):
        from src.basic_apis.metrics_utils.metric_value import calculate
        calculate(self.cfg.calculate_metric_value)

    def plot(self):
        from src.basic_apis.metrics_utils.plot_utils import plot as plot_metric
        plot_metric(self.cfg.plot)

    def dt_collect_dataset(self):
        from src.basic_apis.dt_utils.dataset_collect import collect
        collect(self.cfg.dt_dataset_collect, self.path_manager)

    def dt_collect_dataset_all(self):
        from src.basic_apis.dt_utils.dataset_collect_all import collect
        collect(self.cfg.dt_dataset_collect_all, self.path_manager)

    def dt_collect_dataset_baseline(self):
        from src.basic_apis.dt_baseline.dataset_collect_baseline import collect
        collect(self.cfg.dt_dataset_collect_baseline, self.path_manager)

    def train_ppo_lstm_across_scenario(self):
        from src.basic_apis.ppo.ppo_lstm.train_recurrent_ppo import train as train_lstm_sb3
        train_lstm_sb3(self.cfg.train_ppo_lstm_across_scenario, path_manager=self.path_manager)

    def train_ppo_baseline_across_scenario(self):
        from src.basic_apis.ppo.ppo_baseline.train_ppo_baseline import train_ppo
        train_ppo(self.cfg.train_ppo_baseline_across_scenario, path_manager=self.path_manager)

    def dt_training(self):
        from src.basic_apis.dt_utils.train import train
        train(self.cfg.dt_training, self.path_manager)

    def dt_testing(self):
        from src.basic_apis.dt_utils.test import test_dt_process
        test_dt_process(self.cfg.dt_testing, self.path_manager)

    def train_dt_baseline(self):
        from src.basic_apis.dt_v2.train_dt_baseline import train_hydra
        train_hydra(self.cfg, self.path_manager)

    def test_dt_baseline(self):
        from src.basic_apis.dt_v2.test_dt_baseline import test_dt_baseline
        test_dt_baseline(self.cfg, self.path_manager)

    def train_ppo_oneshot(self):
        from src.basic_apis.ppo.ppo_oneshot.train import train
        train(self.cfg.train_ppo_oneshot, self.path_manager)

    def test_ppo_oneshot(self):
        from src.basic_apis.ppo.ppo_oneshot.test import test_ppo_oneshot
        test_ppo_oneshot(self.cfg.test_ppo_oneshot, self.path_manager)

    def train_ppo_ha(self):
        from src.basic_apis.ppo.ppo_ha.train import train
        train(self.cfg.train_ppo_ha, self.path_manager)

    def test_ppo_ha(self):
        from src.basic_apis.ppo.ppo_ha.test import test_ppo_ha
        test_ppo_ha(self.cfg.test_ppo_ha, self.path_manager)

    def train_ppo_ha_weighted(self):
        from src.basic_apis.ppo.ppo_ha_weighted.train import train
        train(self.cfg.train_ppo_ha_weighted, self.path_manager)

    def test_ppo_ha_weighted(self):
        from src.basic_apis.ppo.ppo_ha_weighted.test import test_ppo_ha
        test_ppo_ha(self.cfg.test_ppo_ha_weighted, self.path_manager)

    def test_ppo_baseline_finetune(self):
        from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_finetune
        test_ppo_baseline_finetune(self.cfg.test_ppo_baseline_finetune, self.path_manager)

    def test_ppo_baseline_multi(self):
        from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_multi
        test_ppo_baseline_multi(self.cfg.test_ppo_baseline_multi, self.path_manager)

    # def plot_hist(self):
    #     from src.basic_apis.metrics_utils.plot_utils import plot_hist
    #     plot_hist(self.cfg)

    def plot_hist(self):
        from src.basic_apis.metrics_utils.plot_hist import plot_hist
        plot_hist(self.cfg.get('plot_hist', None))

    def plot_step(self):
        from src.basic_apis.metrics_utils.plot_step import plot_step
        plot_step(self.cfg.get('plot_step', None))

    def plot_finetune_entropy(self):
        from src.basic_apis.metrics_utils.plot_finetune_entropy import plot_finetune_entropy
        plot_finetune_entropy(self.cfg)

    # === CQL Baseline (M5) ===

    def cql_convert_dataset(self):
        """将 IDT pkl 数据集转换为 d3rlpy MDPDataset (.h5) 格式"""
        from src.basic_apis.cql_baseline.convert_dataset import convert_to_d3rlpy_format
        cfg = self.cfg.cql_convert_dataset
        variants = list(cfg.get('variants', ['expert']))
        for variant in variants:
            input_dir = str(cfg[f'input_{variant}'])
            output_path = str(cfg[f'output_{variant}'])
            print(f"\n[CQL Convert] variant={variant}  {input_dir} → {output_path}")
            convert_to_d3rlpy_format(input_dir, output_path, variant)

    def train_cql(self):
        """训练 CQL Baseline（B1）— 需要先运行 convert_dataset.py 生成 .h5 文件"""
        from src.basic_apis.cql_baseline.train_cql import train_cql
        train_cfg = self.cfg.train_cql
        variants = list(train_cfg.get('variants', ['expert', 'raw']))
        seeds = list(train_cfg.get('train_seeds', [0, 1, 2, 3, 4]))
        asset_cfg = train_cfg.get('asset', None)
        for variant in variants:
            dataset_key = f'dataset_{variant}'
            dataset_path = str(train_cfg.get(dataset_key, ''))
            if not dataset_path:
                print(f"  Skipping {variant}: dataset path not set")
                continue
            for seed in seeds:
                if asset_cfg and bool(asset_cfg.get('use_versioned_runs', False)):
                    model_root = f"{train_cfg.output_root}/{variant}_seed{seed}"
                    run_id_cfg = asset_cfg.get('run_id', 'auto')
                    run_id = None if str(run_id_cfg) == 'auto' else str(run_id_cfg)
                    output_dir = build_versioned_run_dir(model_root, run_id=run_id)
                    if bool(asset_cfg.get('clean_before_run', False)):
                        ensure_clean_dir(output_dir)
                    else:
                        ensure_dir(output_dir)
                else:
                    output_dir = f"{train_cfg.output_root}/{variant}_seed{seed}"
                train_cql(
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    seed=seed,
                    n_steps=int(train_cfg.get('n_steps', 100000)),
                    device=str(train_cfg.get('device', 'cpu')),
                )
                if asset_cfg and bool(asset_cfg.get('use_versioned_runs', False)) and bool(asset_cfg.get('update_latest', True)):
                    model_root = f"{train_cfg.output_root}/{variant}_seed{seed}"
                    link_path = update_latest_symlink(model_root, output_dir)
                    print(f"[Asset] updated latest symlink: {link_path}")

    def test_cql(self):
        """测试 CQL Baseline — 支持 init/max_episode 及 step-level npz 输出"""
        from src.basic_apis.cql_baseline.test_cql import test_cql_scenario
        from omegaconf import OmegaConf
        import os
        import shutil
        test_cfg = self.cfg.test_cql
        env_config = OmegaConf.to_container(self.cfg.environment, resolve=True)
        variants = list(test_cfg.get('variants', ['expert']))
        seeds = list(test_cfg.get('test_seeds', [0, 1, 2, 3, 4]))
        scenarios = list(test_cfg.get('test_scenarios', [5, 6, 7, 8, 9]))
        model_seed_cfg = test_cfg.get('model_seed', None)
        model_seed = int(model_seed_cfg) if model_seed_cfg is not None else None
        model_root = str(test_cfg.get('model_root', 'data/channel_generality/cql_v2/models'))
        save_root = str(test_cfg.get('save_root', 'data/channel_generality/cql_v2'))
        prefer_latest = bool(test_cfg.get('prefer_latest', False))
        clean_before_save = bool(test_cfg.get('clean_before_save', False))
        strict_model_check = bool(test_cfg.get('strict_model_check', True))

        init_ep_cfg = test_cfg.get('init_episode', None)
        max_ep_cfg = test_cfg.get('max_episode', None)
        init_episode = int(init_ep_cfg) if init_ep_cfg is not None else None
        max_episode = int(max_ep_cfg) if max_ep_cfg is not None else None

        if clean_before_save:
            for variant in variants:
                target_dir = os.path.join(save_root, f"cql_{variant}", "metric_json")
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
            print(f"[Asset] cleaned old CQL metrics under: {save_root}")
        for variant in variants:
            for seed in seeds:
                model_seed_for_load = model_seed if model_seed is not None else seed
                if prefer_latest:
                    model_path = f"{model_root}/{variant}_seed{model_seed_for_load}/latest/model.pt"
                else:
                    model_path = f"{model_root}/{variant}_seed{model_seed_for_load}/model.pt"
                if not os.path.exists(model_path):
                    fallback_path = f"{model_root}/{variant}_seed{model_seed_for_load}/model.pt"
                    if os.path.exists(fallback_path):
                        model_path = fallback_path
                if not os.path.exists(model_path):
                    msg = f"CQL model not found: {model_path}"
                    if strict_model_check:
                        raise FileNotFoundError(msg)
                    print(f"[WARN] {msg}, skip variant={variant}, seed={seed}")
                    continue
                for scen in scenarios:
                    test_cql_scenario(
                        model_path=model_path,
                        env_config=env_config,
                        scenario_id=scen,
                        seed=seed,
                        n_episodes=int(test_cfg.get('n_episodes', 100)),
                        variant=variant,
                        save_root=save_root,
                        init_episode=init_episode,
                        max_episode=max_episode,
                    )

    # === Lagrangian PPO (M6) ===

    def train_ppo_lagrangian(self):
        """训练 Lagrangian PPO Baseline（B2）— EnvV2 版"""
        from src.basic_apis.ppo.ppo_lagrangian.train import train_ppo_lagrangian
        train_ppo_lagrangian(self.cfg, self.path_manager)

    def test_ppo_lagrangian(self):
        """测试 Lagrangian PPO Baseline（B2）— EnvV2 版"""
        from src.basic_apis.ppo.ppo_lagrangian.test import test_ppo_lagrangian
        test_ppo_lagrangian(self.cfg, self.path_manager)

    def train_ppo_lagrangian_baseline(self):
        """训练 PPO-Lagrangian Baseline — 与 ppo-baseline/dt-baseline 相同的 CommunicationEnv 环境"""
        from src.basic_apis.ppo.ppo_lagrangian.train_baseline import train_ppo_lagrangian_baseline
        train_ppo_lagrangian_baseline(self.cfg, self.path_manager)

    def test_ppo_lagrangian_baseline(self):
        """测试 PPO-Lagrangian Baseline — s5–s9，ep 0–99，5 个种子"""
        from src.basic_apis.ppo.ppo_lagrangian.test_baseline import test_ppo_lagrangian_baseline
        test_ppo_lagrangian_baseline(self.cfg, self.path_manager)

    # === 新绘图脚本 (M12) ===

    def plot_transfer_curve(self):
        """绘制 Transfer PPO 适配曲线（Fig.9）"""
        from src.basic_apis.metrics_utils.plot_transfer_curve import plot_transfer_curve_from_cfg
        plot_transfer_curve_from_cfg(self.cfg)

    def plot_sensitivity(self):
        """绘制敏感性分析曲线（Fig.10/11）"""
        from src.basic_apis.metrics_utils.plot_sensitivity import plot_sensitivity_from_cfg
        plot_sensitivity_from_cfg(self.cfg)

    # === 推理时间基准测试 (M13) ===

    def inference_benchmark(self):
        """推理时间基准测试（E1 实验）"""
        from src.basic_apis.metrics_utils.inference_benchmark import inference_benchmark
        inference_benchmark(self.cfg, self.path_manager)

    # === 综合结果表生成 (M14) ===

    def generate_results_table(self):
        """生成综合结果 LaTeX 表格（Tab.III / Tab.IV）"""
        from src.basic_apis.metrics_utils.generate_results_table import generate_results_table
        generate_results_table(self.cfg, self.path_manager)

    # === 多 seed 对比绘图 ===

    def plot_multiseed(self):
        """多 seed 对比图（均值 ± 标准差），输出 HP/NHP distance 和 violation 子图。"""
        from src.basic_apis.metrics_utils.plot_s9_multiseed import plot_multiseed_from_cfg
        plot_multiseed_from_cfg(self.cfg)

    def plot_step_multiseed(self):
        """Step-level 时序图（Fig.4/5），从 NPZ 读 step_hp_dist/step_nhp_dist，rolling 平滑后 mean±std。"""
        from src.basic_apis.metrics_utils.plot_s9_multiseed import plot_step_multiseed_from_cfg
        plot_step_multiseed_from_cfg(self.cfg)

    def utils(self):
        from src.basic_apis.general_utils import analyze_assoc_data
        analyze_assoc_data(self.cfg.analyze_assoc_data)

    # === dt_v2 系列：PPO-v2 teacher + DT-v2 ===

    def train_ppo_v2(self):
        """训练 PPO-v2 teacher（EnvV2 + violation reward + 排序动作）。
        使用 SubprocVecEnv + episode 偏移，n_envs=12 最优。
        """
        from src.basic_apis.dt_v2.train_ppo_v2 import train
        train(self.cfg, self.path_manager)

    def test_ppo_v2(self):
        """评测 PPO-v2 模型，使用 held-out ep 60-79（testing 模式）。"""
        import sys, os
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from eval_ppo_v2_per_scenario_expert import test_ppo_v2
        test_ppo_v2(self.cfg, self.path_manager)

    def collect_data_v2(self):
        """用 PPO-v2 teacher 采集 DT 训练数据集（训练 ep 0-59）。"""
        from src.basic_apis.dt_v2.collect_data_v2 import collect
        collect(self.cfg, self.path_manager)

    def train_dt_v2(self):
        """在 PPO-v2 teacher 轨迹上训练 Decision Transformer-v2。"""
        from src.basic_apis.dt_v2.train_dt_v2 import train_hydra
        train_hydra(self.cfg, self.path_manager)

    def test_dt_v2(self):
        """评测 DT-v2，使用 held-out ep 60-79（testing 模式）。"""
        from src.basic_apis.dt_v2.test_v2 import test_dt_v2
        test_dt_v2(self.cfg, self.path_manager)

    def train_dt_v2_tiny(self):
        """训练 DT-v2 tiny 变体（slice_attn + 小主干/小编码器）。"""
        from src.basic_apis.dt_v2.train_dt_v2 import train_hydra_tiny
        train_hydra_tiny(self.cfg, self.path_manager)

    def test_dt_v2_tiny(self):
        """评测 DT-v2 tiny 变体，使用 held-out ep 60-79（testing 模式）。"""
        from src.basic_apis.dt_v2.test_v2 import test_dt_v2_tiny
        test_dt_v2_tiny(self.cfg, self.path_manager)

    # === CQL-v3：多头离散 Q 网络 ===

    def train_cql_v3(self):
        """训练 CQL-v3：多头离散 Q 网络，直接在 MultiDiscrete 动作空间上训练。"""
        from src.basic_apis.cql_baseline.train_cql_discrete import train
        train(self.cfg, self.path_manager)

    def test_cql_v3(self):
        """评测 CQL-v3：使用 HierarchicalSlicingEnvV2，原始离散动作 argmax。"""
        from src.basic_apis.cql_baseline.train_cql_discrete import test
        test(self.cfg, self.path_manager)

