"""Registry mapping Hydra modes to semantic handlers."""

from . import baselines, evaluation, teacher, transfer


CHANNEL_GENERALITY_HANDLERS = {
    # PPO teachers used by objective synthesis and transfer experiments.
    "train_ppo_ha_weighted": teacher.train_ppo_ha_weighted,
    "test_ppo_ha_weighted": teacher.test_ppo_ha_weighted,
    "train_dt_teacher": teacher.train_dt_teacher,
    "test_dt_teacher": teacher.test_dt_teacher,

    # Baselines and controlled solver comparisons.
    "train_ppo_baseline": baselines.train_ppo_baseline,
    "test_ppo_baseline": baselines.test_ppo_baseline,
    "train_ppo_baseline_across_scenario": baselines.train_ppo_baseline_across_scenario,
    "test_ppo_baseline_finetune": baselines.test_ppo_baseline_finetune,
    "test_ppo_baseline_multi": baselines.test_ppo_baseline_multi,
    "train_dt_baseline": baselines.train_dt_baseline,
    "test_dt_baseline": baselines.test_dt_baseline,
    "train_ppo_lagrangian": baselines.train_ppo_lagrangian,
    "test_ppo_lagrangian": baselines.test_ppo_lagrangian,
    "train_ppo_lagrangian_baseline": baselines.train_ppo_lagrangian_baseline,
    "test_ppo_lagrangian_baseline": baselines.test_ppo_lagrangian_baseline,

    # Cross-solver transfer through DT.
    "collect_dt_data": transfer.collect_dt_data,
    "train_dt": transfer.train_dt,
    "test_dt": transfer.test_dt,
    "train_dt_tiny": transfer.train_dt_tiny,
    "test_dt_tiny": transfer.test_dt_tiny,

    # Metrics, plots, and reporting.
    "calculate_metric_value": evaluation.calculate_metric_value,
    "plot": evaluation.plot,
    "plot_s9": evaluation.plot,
    "plot_hist": evaluation.plot_hist,
    "plot_step": evaluation.plot_step,
    "plot_finetune_entropy": evaluation.plot_finetune_entropy,
    "plot_transfer_curve": evaluation.plot_transfer_curve,
    "plot_sensitivity": evaluation.plot_sensitivity,
    "inference_benchmark": evaluation.inference_benchmark,
    "generate_results_table": evaluation.generate_results_table,
    "summarize_mvp": evaluation.summarize_mvp,
    "plot_multiseed": evaluation.plot_multiseed,
    "plot_step_multiseed": evaluation.plot_step_multiseed,
    "analyze_assoc_data": evaluation.analyze_assoc_data,
}
