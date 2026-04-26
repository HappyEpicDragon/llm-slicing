"""Metric, plotting, and reporting handlers."""


def calculate_metric_value(sim):
    from src.basic_apis.metrics_utils.metric_value import calculate

    calculate(sim.cfg.calculate_metric_value)


def plot(sim):
    from src.basic_apis.metrics_utils.plot_utils import plot as plot_metric

    plot_metric(sim.cfg.plot)


def plot_hist(sim):
    from src.basic_apis.metrics_utils.plot_hist import plot_hist

    plot_hist(sim.cfg.get("plot_hist", None))


def plot_step(sim):
    from src.basic_apis.metrics_utils.plot_step import plot_step

    plot_step(sim.cfg.get("plot_step", None))


def plot_finetune_entropy(sim):
    from src.basic_apis.metrics_utils.plot_finetune_entropy import plot_finetune_entropy

    plot_finetune_entropy(sim.cfg)


def plot_transfer_curve(sim):
    from src.basic_apis.metrics_utils.plot_transfer_curve import plot_transfer_curve_from_cfg

    plot_transfer_curve_from_cfg(sim.cfg)


def plot_sensitivity(sim):
    from src.basic_apis.metrics_utils.plot_sensitivity import plot_sensitivity_from_cfg

    plot_sensitivity_from_cfg(sim.cfg)


def inference_benchmark(sim):
    from src.basic_apis.metrics_utils.inference_benchmark import inference_benchmark

    inference_benchmark(sim.cfg)


def generate_results_table(sim):
    from src.basic_apis.metrics_utils.generate_results_table import generate_results_table

    generate_results_table(sim.cfg)


def plot_multiseed(sim):
    from src.basic_apis.metrics_utils.plot_s9_multiseed import plot_multiseed_from_cfg

    plot_multiseed_from_cfg(sim.cfg)


def plot_step_multiseed(sim):
    from src.basic_apis.metrics_utils.plot_s9_multiseed import plot_step_multiseed_from_cfg

    plot_step_multiseed_from_cfg(sim.cfg)


def analyze_assoc_data(sim):
    from src.basic_apis.utils.analysis import analyze_assoc_data as run_assoc_data_analysis

    run_assoc_data_analysis(sim.cfg.analyze_assoc_data)
