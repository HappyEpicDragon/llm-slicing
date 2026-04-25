from os import getcwd
from pathlib import Path
from typing import Any, Optional


def _read_cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class PathContext:
    """Resolved filesystem paths shared by simulation components."""

    def __init__(
        self,
        hydra_workdir: str,
        paths_cfg: Optional[Any] = None,
        project_root: Optional[str] = None,
    ):
        self.hydra_workdir = str(hydra_workdir)
        self.project_root = str(
            project_root
            or _read_cfg(paths_cfg, "project_root")
            or getcwd()
        )
        self.root_path = self.project_root

        data_root = _read_cfg(paths_cfg, "data_root")
        static_root = _read_cfg(paths_cfg, "static_root")
        self.data_root = str(data_root or Path(self.project_root) / "data")
        self.static_root = str(static_root or Path(self.data_root) / "static")
        self.association_root = str(
            _read_cfg(paths_cfg, "association_root")
            or Path(self.static_root) / "association"
        )
        self.channel_root = str(
            _read_cfg(paths_cfg, "channel_root")
            or Path(self.static_root) / "channel"
        )
        self.channel_file_name = str(
            _read_cfg(paths_cfg, "channel_file_name", "target_cell_power.mat")
        )
        self.channel_npz_name = str(
            _read_cfg(paths_cfg, "channel_npz_name", "target_cell_power.npz")
        )
        self.runtime_metrics_subdir = str(
            _read_cfg(paths_cfg, "runtime_metrics_subdir", "")
        )
        self.ray_results_dirname = str(
            _read_cfg(paths_cfg, "ray_results_dirname", "ray_results")
        )

    @classmethod
    def from_cfg(
        cls,
        cfg: Any,
        hydra_workdir: Optional[str] = None,
        project_root: Optional[str] = None,
    ) -> "PathContext":
        paths_cfg = _read_cfg(cfg, "paths", cfg)
        workdir = hydra_workdir or _read_cfg(cfg, "workdir") or getcwd()
        return cls(workdir, paths_cfg=paths_cfg, project_root=project_root)

    def get_channel_file_path(self, association_to_use, episode_to_use):
        return str(
            Path(self.channel_root)
            / f"assoc_{association_to_use}"
            / f"ep_{episode_to_use}"
            / self.channel_file_name
        )

    def get_channel_npz_path(self, association_to_use, episode_to_use):
        return str(
            Path(self.channel_root)
            / f"assoc_{association_to_use}"
            / f"ep_{episode_to_use}"
            / self.channel_npz_name
        )

    def get_save_metrics_dir_path(self):
        if not self.runtime_metrics_subdir:
            return self.hydra_workdir
        return str(Path(self.hydra_workdir) / self.runtime_metrics_subdir)

    def get_read_checkpoint_path(self):
        return str(Path(self.hydra_workdir) / self.ray_results_dirname)

    def get_association_file_path(self, episode_number):
        return str(Path(self.association_root) / f"ep_{episode_number}.npz")


def create_path_context(
    cfg: Any,
    hydra_workdir: Optional[str] = None,
    project_root: Optional[str] = None,
) -> PathContext:
    return PathContext.from_cfg(
        cfg,
        hydra_workdir=hydra_workdir,
        project_root=project_root,
    )
