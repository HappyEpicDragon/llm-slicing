import os
import re
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.simulation.base_simu.simulation_factory import SimulationFactory
from src.simulation.base_simu.base_simulation import BaseSimulation


def load_reward_fn_from_file(path: str):
    """从 ReEvo 导出的 .txt 中解析 ```python ... ``` 代码块并返回 compute_reward* 函数。"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"```python\s*\n(.*?)```", content, re.DOTALL)
    code = match.group(1) if match else content
    namespace = {}
    exec(code, namespace)
    candidates = []
    for name, obj in namespace.items():
        if callable(obj) and name.startswith("compute_reward"):
            candidates.append((name, obj))
    if not candidates:
        raise ValueError(f"No compute_reward function found in {path}")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


@SimulationFactory.register("evo")
class EvoSimulator(BaseSimulation):
    """EvoSimulator：负责 ReEvo 进化搜索相关逻辑。

    支持两种 mode：
        run_evo       — 启动 ReEvo 进化搜索（PPO proxy 评估）
        eval_candidate — 对指定 reward function 运行完整 PPO 训练（可选测试）
    """

    def run(self):
        match self.cfg.mode:
            case "run_evo":
                self.run_evo()
            case "eval_candidate":
                self.eval_candidate()
            case _:
                raise ValueError(f"Unknown mode: {self.cfg.mode}")

    # ------------------------------------------------------------------
    # mode: run_evo
    # ------------------------------------------------------------------
    def run_evo(self):
        """直接调用 ReEvo 进化搜索（单层 Hydra，无 subprocess）。"""
        import hydra as _hydra
        from datetime import datetime
        from src.reevo.utils.utils import init_client
        from src.reevo.reevo import ReEvo

        evo_cfg = self.cfg.run_evo
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        reevo_root = os.path.join(project_root, "src", "reevo")

        # 构建 ReEvo config（深拷贝，避免修改原始 cfg）
        reevo_cfg = OmegaConf.create(
            OmegaConf.to_container(evo_cfg.reevo, resolve=True)
        )
        # 用 proxy_scenario 覆盖 problem_size（演化搜索场景选择）
        reevo_cfg.problem.problem_size = str(evo_cfg.proxy_scenario)

        # 创建输出目录（替代 Hydra #2 的 chdir 行为）
        problem_name = reevo_cfg.problem.problem_name
        problem_type = reevo_cfg.problem.problem_type
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        reevo_outputs = str(self.cfg.get("paths", {}).get("reevo_outputs", os.path.join(project_root, "outputs", "runs", "reevo")))
        output_dir = os.path.join(reevo_outputs, f"{problem_name}-{problem_type}", ts)
        os.makedirs(output_dir, exist_ok=True)

        # 设置 eval 子进程所需的环境变量
        os.environ["IDT_PROJECT_ROOT"] = project_root
        os.environ["IDT_PROXY_TRAIN_RATIO"] = str(evo_cfg.proxy_train_ratio)
        os.environ["IDT_SEED"] = str(evo_cfg.seed)
        proxy_scenarios = str(evo_cfg.get("proxy_scenarios", ""))
        if proxy_scenarios:
            os.environ["IDT_PROXY_SCENARIOS"] = proxy_scenarios

        # 初始化 LLM 客户端
        client = init_client(reevo_cfg)
        long_ref_llm = _hydra.utils.instantiate(reevo_cfg.llm_long_ref) if reevo_cfg.get("llm_long_ref") else None
        short_ref_llm = _hydra.utils.instantiate(reevo_cfg.llm_short_ref) if reevo_cfg.get("llm_short_ref") else None
        crossover_llm = _hydra.utils.instantiate(reevo_cfg.llm_crossover) if reevo_cfg.get("llm_crossover") else None
        mutation_llm = _hydra.utils.instantiate(reevo_cfg.llm_mutation) if reevo_cfg.get("llm_mutation") else None

        # chdir 到输出目录（ReEvo 所有文件 IO 依赖 CWD）
        old_cwd = os.getcwd()
        os.chdir(output_dir)
        print(f"[EvoSimulator] ReEvo output dir: {output_dir}")
        print(f"[EvoSimulator] reevo_root: {reevo_root}")
        try:
            lhh = ReEvo(
                reevo_cfg, reevo_root, client,
                long_reflector_llm=long_ref_llm,
                short_reflector_llm=short_ref_llm,
                crossover_llm=crossover_llm,
                mutation_llm=mutation_llm,
            )
            best_code, best_code_path = lhh.evolve()

            # 验证最佳代码
            with open(os.path.join(reevo_root, "problems", problem_name, "gpt.py"), "w", encoding="utf-8") as f:
                f.write(best_code + "\n")
            test_script = os.path.join(reevo_root, "problems", problem_name, "eval.py")
            val_stdout_path = "best_code_overall_val_stdout.txt"
            print(f"[EvoSimulator] Running validation: {test_script}")
            with open(val_stdout_path, "w", encoding="utf-8") as stdout_f:
                subprocess.run(
                    [sys.executable, test_script, "-1", reevo_root, "val"],
                    stdout=stdout_f, stderr=stdout_f,
                )
            with open(val_stdout_path, "r", encoding="utf-8") as f:
                for line in f:
                    print(f"[EvoSimulator] val: {line.rstrip()}")

            best_response_path = best_code_path.replace(".py", ".txt").replace("code", "response")
            print(f"[EvoSimulator] Evolution complete. Best: {best_response_path}")
        finally:
            os.chdir(old_cwd)

    # ------------------------------------------------------------------
    # mode: eval_candidate
    # ------------------------------------------------------------------
    def eval_candidate(self):
        """加载指定 reward，在各场景上全量 PPO 训练，保存模型；可选跑 test_ppo_ha。"""
        from src.basic_apis.ppo.ppo_ha_weighted.train import train
        from src.basic_apis.ppo.ppo_ha_weighted.test import test_ppo_ha

        cand = self.cfg.eval_candidate
        reward_fn_path = str(cand.reward_fn_path)
        scenarios = list(cand.scenarios)
        seed = int(cand.seed)
        model_save_root = str(cand.model_save_root)
        run_test = bool(cand.run_test)
        test_seeds = list(cand.test_seeds)
        test_save_root = str(cand.test_save_root)
        train_timesteps = getattr(cand, "train_timesteps", None)

        project_root = str(Path(__file__).resolve().parent.parent.parent)
        conf_dir = os.path.join(project_root, "conf")

        reward_fn = load_reward_fn_from_file(reward_fn_path)
        print(f"[EvoSimulator] eval_candidate reward_fn_path={reward_fn_path}")
        print(f"[EvoSimulator] scenarios={scenarios}, seed={seed}, run_test={run_test}")

        train_rows = []
        test_rows = []

        for scenario in scenarios:
            sc = int(scenario)
            model_root_scenario = os.path.join(model_save_root, f"scenario_{sc}")
            tmp_dir = tempfile.mkdtemp(prefix="evo_eval_cand_")
            try:
                GlobalHydra.instance().clear()
                with initialize_config_dir(
                    config_dir=conf_dir, job_name="evo_eval_candidate_train", version_base=None
                ):
                    overrides = [
                        "simulation=channel_generality/train_ppo_ha_weighted",
                        f"train_scenario={sc}",
                        f"workdir={tmp_dir}",
                        f"train_ppo_ha_weighted.asset.model_root={model_root_scenario}",
                        "train_ppo_ha_weighted.asset.update_latest=false",
                    ]
                    cfg = compose(
                        config_name="conf",
                        overrides=overrides,
                    )
                    train_cfg = cfg.train_ppo_ha_weighted
                    train_cfg_dict = OmegaConf.to_container(train_cfg, resolve=True)
                    if train_timesteps is not None:
                        train_cfg_dict["environment"]["train_rl"]["total_timesteps"] = int(train_timesteps)
                train_cfg_resolved = OmegaConf.create(train_cfg_dict)

                metrics = train(
                    train_cfg_resolved,
                    paths_cfg=cfg.paths,
                    workdir=tmp_dir,
                    reward_fn=reward_fn,
                    seed=seed,
                )
                hp_r = float(metrics.get("hp_violation_rate", float("nan")))
                nhp_r = float(metrics.get("nhp_violation_rate", float("nan")))
                train_rows.append((sc, hp_r, nhp_r))
                print(
                    f"[EvoSimulator] train scenario={sc} hp_violation_rate={hp_r:.6f} "
                    f"nhp_violation_rate={nhp_r:.6f}"
                )

                runs_dir = os.path.join(model_root_scenario, "runs")
                run_entries = sorted(Path(runs_dir).iterdir(), key=lambda p: p.name, reverse=True)
                if not run_entries:
                    raise RuntimeError(f"No versioned runs found in {runs_dir} after training")
                model_path = str(run_entries[0] / "best_model" / "best_model.zip")

                if run_test:
                    GlobalHydra.instance().clear()
                    test_tmp = tempfile.mkdtemp(prefix="evo_eval_cand_test_")
                    try:
                        seeds_csv = ",".join(str(int(s)) for s in test_seeds)
                        with initialize_config_dir(
                            config_dir=conf_dir, job_name="evo_eval_candidate_test", version_base=None
                        ):
                            tcfg = compose(
                                config_name="conf",
                                overrides=[
                                    "simulation=channel_generality/test_ppo_ha_weighted",
                                    f"workdir={test_tmp}",
                                    f"test_ppo_ha_weighted.model_path={model_path}",
                                    f"test_ppo_ha_weighted.test_seeds=[{seeds_csv}]",
                                    f"test_ppo_ha_weighted.save_root={test_save_root}",
                                    f"test_ppo_ha_weighted.env_updates.model_name=scenario_{sc}",
                                    f"test_ppo_ha_weighted.env_updates.inside.training.active_scenario_list=[{sc}]",
                                    f"test_ppo_ha_weighted.env_updates.inside.evaluating.active_scenario_list=[{sc}]",
                                    f"test_ppo_ha_weighted.env_updates.inside.testing.active_scenario_list=[{sc}]",
                                ],
                            )
                            test_block = tcfg.test_ppo_ha_weighted
                            test_cfg_dict = OmegaConf.to_container(test_block, resolve=True)
                        test_cfg_resolved = OmegaConf.create(test_cfg_dict)

                        test_out = test_ppo_ha(
                            test_cfg_resolved,
                            paths_cfg=tcfg.paths,
                            workdir=test_tmp,
                        )
                        test_rows.append((sc, test_out))
                        print(f"[EvoSimulator] test scenario={sc} summary keys={list(test_out.keys())}")
                    finally:
                        shutil.rmtree(test_tmp, ignore_errors=True)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        print("\n[EvoSimulator] ========== eval_candidate 汇总 ==========")
        print("训练（TtiViolationRateCallback 率）:")
        for sc, hp_r, nhp_r in train_rows:
            print(f"  scenario {sc}: hp_violation_rate={hp_r:.6f}, nhp_violation_rate={nhp_r:.6f}")
        if run_test and test_rows:
            print("测试（test_ppo_ha 各场景 summary）:")
            for sc, out in test_rows:
                key = f"scenario_{sc}"
                if key in out:
                    print(f"  scenario {sc}: {out[key]}")
                else:
                    print(f"  scenario {sc}: {out}")
