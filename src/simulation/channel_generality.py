from .base_simu.simulation_factory import SimulationFactory
from .base_simu.base_simulation import BaseSimulation
from hydra.core.hydra_config import HydraConfig

from src.basic_apis.asset_utils import build_versioned_run_dir, update_latest_symlink, ensure_clean_dir, ensure_dir


@SimulationFactory.register("channel_generality")
class ChannelGeneralitySimulator(BaseSimulation):
    def __init__(self, cfg):
        super().__init__(cfg)
        hydra_run_output_dir = HydraConfig.get().runtime.output_dir
        self.workdir = hydra_run_output_dir
        self.paths_cfg = self.cfg.paths

    def run(self):
        print("Current Simulation: Channel Generality")
        print(f"Current mode: {self.cfg.mode}")
        print("=" * 60 + f"\n")

        match self.cfg.mode:
            case "train_ppo_baseline":
                self.train_ppo_baseline()

            case "test_ppo_baseline":
                self.test_ppo_baseline()

            case "calculate_metric_value":
                self.calculate_metric_value()

            case "plot":
                self.plot()

            case "train_ppo_baseline_across_scenario":
                self.train_ppo_baseline_across_scenario()

            case "train_ppo_ha_weighted":
                self.train_ppo_ha_weighted()

            case "train_dt_baseline":
                self.train_dt_baseline()

            case "test_dt_baseline":
                self.test_dt_baseline()

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

            case _:
                raise ValueError(f"未知的模式: {self.cfg.mode}")

    def train_ppo_baseline(self):
        """训练基线 PPO"""
        from src.basic_apis.ppo.ppo_baseline.train_ppo_baseline import train_ppo
        train_ppo(self.cfg.train_ppo_baseline, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def test_ppo_baseline(self):
        from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline
        test_ppo_baseline(self.cfg.test_ppo_baseline, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def calculate_metric_value(self):
        from src.basic_apis.metrics_utils.metric_value import calculate
        calculate(self.cfg.calculate_metric_value)

    def plot(self):
        from src.basic_apis.metrics_utils.plot_utils import plot as plot_metric
        plot_metric(self.cfg.plot)

    def train_ppo_baseline_across_scenario(self):
        from src.basic_apis.ppo.ppo_baseline.train_ppo_baseline import train_ppo
        train_ppo(self.cfg.train_ppo_baseline_across_scenario, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def train_dt_baseline(self):
        from src.basic_apis.dt_v2.train_dt_baseline import train_hydra
        train_hydra(self.cfg)

    def test_dt_baseline(self):
        from src.basic_apis.dt_v2.test_dt_baseline import test_dt_baseline
        test_dt_baseline(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def train_ppo_ha_weighted(self):
        from src.basic_apis.ppo.ppo_ha_weighted.train import train
        train(
            self.cfg.train_ppo_ha_weighted,
            paths_cfg=self.paths_cfg,
            workdir=self.workdir,
        )

    def test_ppo_ha_weighted(self):
        from src.basic_apis.ppo.ppo_ha_weighted.test import test_ppo_ha
        test_ppo_ha(
            self.cfg.test_ppo_ha_weighted,
            paths_cfg=self.paths_cfg,
            workdir=self.workdir,
        )

    def test_ppo_baseline_finetune(self):
        from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_finetune
        test_ppo_baseline_finetune(self.cfg.test_ppo_baseline_finetune, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def test_ppo_baseline_multi(self):
        from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_multi
        test_ppo_baseline_multi(self.cfg.test_ppo_baseline_multi, paths_cfg=self.paths_cfg, workdir=self.workdir)

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


    # === Lagrangian PPO (M6) ===

    def train_ppo_lagrangian(self):
        """训练 Lagrangian PPO Baseline（B2）— EnvV2 版"""
        from src.basic_apis.ppo.ppo_lagrangian.train import train_ppo_lagrangian
        train_ppo_lagrangian(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def test_ppo_lagrangian(self):
        """测试 Lagrangian PPO Baseline（B2）— EnvV2 版"""
        from src.basic_apis.ppo.ppo_lagrangian.test import test_ppo_lagrangian
        test_ppo_lagrangian(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def train_ppo_lagrangian_baseline(self):
        """训练 PPO-Lagrangian Baseline — 与 ppo-baseline/dt-baseline 相同的 CommunicationEnv 环境"""
        from src.basic_apis.ppo.ppo_lagrangian.train_baseline import train_ppo_lagrangian_baseline
        train_ppo_lagrangian_baseline(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def test_ppo_lagrangian_baseline(self):
        """测试 PPO-Lagrangian Baseline — s5–s9，ep 0–99，5 个种子"""
        from src.basic_apis.ppo.ppo_lagrangian.test_baseline import test_ppo_lagrangian_baseline
        test_ppo_lagrangian_baseline(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

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
        inference_benchmark(self.cfg)

    # === 综合结果表生成 (M14) ===

    def generate_results_table(self):
        """生成综合结果 LaTeX 表格（Tab.III / Tab.IV）"""
        from src.basic_apis.metrics_utils.generate_results_table import generate_results_table
        generate_results_table(self.cfg)

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
        train(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def test_ppo_v2(self):
        """评测 PPO-v2 模型，使用 held-out ep 60-79（testing 模式）。"""
        import sys, os
        scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from eval_ppo_v2_per_scenario_expert import test_ppo_v2
        test_ppo_v2(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def collect_data_v2(self):
        """用 PPO-v2 teacher 采集 DT 训练数据集（训练 ep 0-59）。"""
        from src.basic_apis.dt_v2.collect_data_v2 import collect
        collect(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def train_dt_v2(self):
        """在 PPO-v2 teacher 轨迹上训练 Decision Transformer-v2。"""
        from src.basic_apis.dt_v2.train_dt_v2 import train_hydra
        train_hydra(self.cfg)

    def test_dt_v2(self):
        """评测 DT-v2，使用 held-out ep 60-79（testing 模式）。"""
        from src.basic_apis.dt_v2.test_v2 import test_dt_v2
        test_dt_v2(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)

    def train_dt_v2_tiny(self):
        """训练 DT-v2 tiny 变体（slice_attn + 小主干/小编码器）。"""
        from src.basic_apis.dt_v2.train_dt_v2 import train_hydra_tiny
        train_hydra_tiny(self.cfg)

    def test_dt_v2_tiny(self):
        """评测 DT-v2 tiny 变体，使用 held-out ep 60-79（testing 模式）。"""
        from src.basic_apis.dt_v2.test_v2 import test_dt_v2_tiny
        test_dt_v2_tiny(self.cfg, paths_cfg=self.paths_cfg, workdir=self.workdir)


