from .base_simu.simulation_factory import SimulationFactory
from .base_simu.base_simulation import BaseSimulation
from hydra.core.hydra_config import HydraConfig

from src.simulation.channel_generality_modes import CHANNEL_GENERALITY_HANDLERS


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

        handler = CHANNEL_GENERALITY_HANDLERS.get(self.cfg.mode)
        if handler is None:
            raise ValueError(f"未知的模式: {self.cfg.mode}")
        handler(self)
