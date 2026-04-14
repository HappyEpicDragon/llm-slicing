from .base_simu.simulation_factory import SimulationFactory
from .base_simu.base_simulation import BaseSimulation
from hydra.core.hydra_config import HydraConfig

from src.basic_apis.network_slicing_business.path_manager import PathManager


@SimulationFactory.register("explainable_ai")
class ExplainableAISimulator(BaseSimulation):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 将 path_manager 作为实例变量
        hydra_run_output_dir = HydraConfig.get().runtime.output_dir
        self.path_manager = PathManager(hydra_run_output_dir)

    def run(self):
        print("Current Simulation: Explainable AI")
        print(f"Current mode: {self.cfg.mode}")
        print("=" * 60 + f"\n")

        match self.cfg.mode:
            case "semantic_manifold":
                self.semantic_manifold()

            case "attention":
                self.attention()

            case "attention_event_triggered":
                self.attention_event_triggered()

            case "rtg_sweeping":
                self.rtg_sweeping()



    def semantic_manifold(self):
        from src.basic_apis.explainable_ai_utils.semantic_manifold.ppo import run_ppo_collection
        from src.basic_apis.explainable_ai_utils.semantic_manifold.dt import run_dt_collection
        from src.basic_apis.explainable_ai_utils.semantic_manifold.visualize import visualize
        run_ppo_collection(self.cfg.semantic_manifold, self.path_manager)
        run_dt_collection(self.cfg.semantic_manifold, self.path_manager)
        visualize(self.cfg.semantic_manifold, self.path_manager)

    def attention(self):
        from src.basic_apis.explainable_ai_utils.attention.attention import execute
        execute(self.cfg.attention, self.path_manager, self.cfg.environment)

    def attention_event_triggered(self):
        from src.basic_apis.explainable_ai_utils.attention.event_triggered import execute
        execute(self.cfg.attention_event_triggered, self.path_manager, self.cfg.environment)

    def rtg_sweeping(self):
        from src.basic_apis.explainable_ai_utils.rtg_sweeping.rtg_sweeping import execute
        execute(self.cfg.rtg_sweeping, self.path_manager, self.cfg.environment)
