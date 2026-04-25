import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra as _hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from src.basic_apis.asset_utils import build_versioned_run_dir, ensure_dir, update_latest_symlink
from src.simulation.base_simu.base_simulation import BaseSimulation
from src.simulation.base_simu.simulation_factory import SimulationFactory


log = logging.getLogger(__name__)


def _to_plain_dict(cfg_section):
    return OmegaConf.to_container(cfg_section, resolve=True)


def _write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@SimulationFactory.register("power_control")
class PowerControlSimulator(BaseSimulation):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.workdir = Path(HydraConfig.get().runtime.output_dir)

    def run(self):
        print("Current Simulation: Power Control")
        print(f"Current mode: {self.cfg.mode}")
        print("=" * 60 + "\n")

        match self.cfg.mode:
            case "train_ppo_discrete":
                self.train_ppo_discrete()
            case "test_ppo_discrete":
                self.test_ppo_discrete()
            case "run_evo":
                self.run_evo()
            case _:
                raise ValueError(f"Unknown mode: {self.cfg.mode}")

    def train_ppo_discrete(self):
        from src.basic_apis.power_control.train import train

        cfg = self.cfg.train_ppo_discrete
        env_config = _to_plain_dict(cfg.environment)
        ppo_config = _to_plain_dict(cfg.ppo)
        asset_cfg = cfg.asset

        model_root = Path(asset_cfg.model_root)
        run_id = None if str(asset_cfg.run_id) == "auto" else str(asset_cfg.run_id)
        run_dir = Path(build_versioned_run_dir(str(model_root), run_id))
        ensure_dir(str(run_dir))
        save_path = run_dir / cfg.output.model_name

        metrics = train(
            env_config=env_config,
            ppo_config=ppo_config,
            total_timesteps=int(cfg.training.total_timesteps),
            save_path=save_path,
            reward_fn=None,
            seed=int(cfg.training.seed),
            verbose=int(cfg.training.verbose),
        )

        metrics_payload = {
            "mode": self.cfg.mode,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "model_path": str(save_path),
            "environment": env_config,
            "ppo": ppo_config,
            "training": _to_plain_dict(cfg.training),
            "metrics": metrics,
        }
        _write_json(Path(cfg.output.results_path) / "metrics.json", metrics_payload)

        latest_link = None
        if bool(asset_cfg.update_latest):
            latest_link = update_latest_symlink(str(model_root), str(run_dir))
            metrics_payload["latest_link"] = latest_link
            _write_json(Path(cfg.output.results_path) / "metrics.json", metrics_payload)

        print(f"Training metrics saved to {Path(cfg.output.results_path) / 'metrics.json'}")
        print(f"Model run dir: {run_dir}")
        if latest_link is not None:
            print(f"Latest model link updated: {latest_link}")

    def test_ppo_discrete(self):
        from src.basic_apis.power_control.train import load_and_evaluate

        cfg = self.cfg.test_ppo_discrete
        env_config = _to_plain_dict(cfg.environment)
        metrics = load_and_evaluate(
            model_path=Path(str(cfg.testing.model_path)),
            env_config=env_config,
            n_episodes=int(cfg.testing.n_eval_episodes),
            seed=int(cfg.testing.seed),
        )

        results_payload = {
            "mode": self.cfg.mode,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "model_path": str(cfg.testing.model_path),
            "environment": env_config,
            "testing": _to_plain_dict(cfg.testing),
            "metrics": metrics,
        }
        _write_json(Path(cfg.output.results_path) / "metrics.json", results_payload)
        print(f"Test metrics saved to {Path(cfg.output.results_path) / 'metrics.json'}")

    def run_evo(self):
        from src.reevo.reevo import ReEvo
        from src.reevo.utils.utils import init_client

        cfg = self.cfg.run_evo
        project_root = str(Path(__file__).resolve().parent.parent.parent)
        reevo_root = os.path.join(project_root, "src", "reevo")

        reevo_cfg = OmegaConf.create(OmegaConf.to_container(cfg.reevo, resolve=True))
        reevo_cfg.problem.problem_size = str(cfg.proxy_scenario)

        output_dir = Path(cfg.output_dir)
        ensure_dir(str(output_dir))

        old_cwd = os.getcwd()
        old_env = {
            "PC_PROJECT_ROOT": os.environ.get("PC_PROJECT_ROOT"),
            "PC_PROXY_TRAIN_RATIO": os.environ.get("PC_PROXY_TRAIN_RATIO"),
            "PC_SEED": os.environ.get("PC_SEED"),
        }
        os.environ["PC_PROJECT_ROOT"] = project_root
        os.environ["PC_PROXY_TRAIN_RATIO"] = str(cfg.proxy_train_ratio)
        os.environ["PC_SEED"] = str(cfg.seed)

        client = init_client(reevo_cfg)
        long_ref_llm = _hydra.utils.instantiate(reevo_cfg.llm_long_ref) if reevo_cfg.get("llm_long_ref") else None
        short_ref_llm = _hydra.utils.instantiate(reevo_cfg.llm_short_ref) if reevo_cfg.get("llm_short_ref") else None
        crossover_llm = _hydra.utils.instantiate(reevo_cfg.llm_crossover) if reevo_cfg.get("llm_crossover") else None
        mutation_llm = _hydra.utils.instantiate(reevo_cfg.llm_mutation) if reevo_cfg.get("llm_mutation") else None

        os.chdir(output_dir)
        print(f"[PowerControlSimulator] ReEvo output dir: {output_dir}")
        try:
            reevo = ReEvo(
                reevo_cfg,
                reevo_root,
                client,
                long_reflector_llm=long_ref_llm,
                short_reflector_llm=short_ref_llm,
                crossover_llm=crossover_llm,
                mutation_llm=mutation_llm,
            )
            best_code, best_code_path = reevo.evolve()
        finally:
            os.chdir(old_cwd)
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        summary = {
            "mode": self.cfg.mode,
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "project_root": project_root,
            "reevo_root": reevo_root,
            "proxy_scenario": int(cfg.proxy_scenario),
            "proxy_train_ratio": float(cfg.proxy_train_ratio),
            "seed": int(cfg.seed),
            "output_dir": str(output_dir),
            "best_code_path": str(best_code_path),
        }
        if best_code is not None:
            summary["best_code_preview"] = "\n".join(best_code.splitlines()[:20])
        _write_json(output_dir / "summary.json", summary)
        print(f"Evolution summary saved to {output_dir / 'summary.json'}")
