from os import getcwd


class PathManager:
    """
    Manages paths for various data and result files used in the simulation.

    Attributes:
        root_path (str): The root directory for all simulation-related files.
    """

    def __init__(
            self,
            hydra_workdir: str,
    ):
        self.root_path = getcwd()
        self.hydra_workdir = hydra_workdir

    def get_channel_file_path(self, association_to_use, episode_to_use):
        return f"{self.root_path}/data/static/channel/assoc_{association_to_use}/ep_{episode_to_use}/target_cell_power.mat"

    def get_channel_npz_path(self, association_to_use, episode_to_use):
        return f"{self.root_path}/data/static/channel/assoc_{association_to_use}/ep_{episode_to_use}/target_cell_power.npz"

    # 2. sixg_radio_mgmt/metrics.py
    def get_save_metrics_dir_path(self):
        # path = f"{self.hydra_workdir}/hist/"
        # path = f"{self.root_path}/data/channel_generality/ppo_baseline/metric_raw"
        path = f"{self.hydra_workdir}"
        return path

    # 4. ray_agent.py
    def get_read_checkpoint_path(self):
        return f"{self.hydra_workdir}/ray_results"

    # association
    def get_association_file_path(self, episode_number):
        return f"{self.root_path}/data/static/association/ep_{episode_number}.npz"