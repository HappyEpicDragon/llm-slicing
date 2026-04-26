from src.simulation.base_simu.base_simulation import BaseSimulation
from src.simulation.base_simu.simulation_factory import SimulationFactory


@SimulationFactory.register("orchestration")
class OrchestrationSimulator(BaseSimulation):
    """Experiment-group orchestration entrypoints."""

    def run(self):
        match self.cfg.mode:
            case "run_mvp_agentic_s0":
                from src.basic_apis.orchestration.tmux_mvp import run_mvp_agentic_s0

                run_mvp_agentic_s0(self.cfg)
            case _:
                raise ValueError(f"Unknown orchestration mode: {self.cfg.mode}")
