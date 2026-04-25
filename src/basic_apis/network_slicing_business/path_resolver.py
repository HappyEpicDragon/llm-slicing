from os import getcwd
from pathlib import Path
from typing import Any, Optional


def read_cfg(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def project_root(paths_cfg: Any = None, explicit_root: Optional[str] = None) -> str:
    return str(explicit_root or read_cfg(paths_cfg, "project_root") or getcwd())


def data_root(paths_cfg: Any = None, explicit_root: Optional[str] = None) -> str:
    configured = read_cfg(paths_cfg, "data_root")
    return str(configured or Path(project_root(paths_cfg, explicit_root)) / "data")


def static_root(paths_cfg: Any = None, explicit_root: Optional[str] = None) -> str:
    configured = read_cfg(paths_cfg, "static_root")
    return str(configured or Path(data_root(paths_cfg, explicit_root)) / "static")


def association_root(paths_cfg: Any = None, explicit_root: Optional[str] = None) -> str:
    configured = read_cfg(paths_cfg, "association_root")
    return str(configured or Path(static_root(paths_cfg, explicit_root)) / "association")


def channel_root(paths_cfg: Any = None, explicit_root: Optional[str] = None) -> str:
    configured = read_cfg(paths_cfg, "channel_root")
    return str(configured or Path(static_root(paths_cfg, explicit_root)) / "channel")


def channel_file_name(paths_cfg: Any = None) -> str:
    return str(read_cfg(paths_cfg, "channel_file_name", "target_cell_power.mat"))


def channel_npz_name(paths_cfg: Any = None) -> str:
    return str(read_cfg(paths_cfg, "channel_npz_name", "target_cell_power.npz"))


def runtime_metrics_subdir(paths_cfg: Any = None) -> str:
    return str(read_cfg(paths_cfg, "runtime_metrics_subdir", ""))


def ray_results_dirname(paths_cfg: Any = None) -> str:
    return str(read_cfg(paths_cfg, "ray_results_dirname", "ray_results"))


def association_file_path(paths_cfg: Any, episode_number, explicit_root: Optional[str] = None) -> str:
    return str(Path(association_root(paths_cfg, explicit_root)) / f"ep_{episode_number}.npz")


def channel_mat_path(
    paths_cfg: Any,
    association_to_use,
    episode_to_use,
    explicit_root: Optional[str] = None,
) -> str:
    return str(
        Path(channel_root(paths_cfg, explicit_root))
        / f"assoc_{association_to_use}"
        / f"ep_{episode_to_use}"
        / channel_file_name(paths_cfg)
    )


def channel_npz_path(
    paths_cfg: Any,
    association_to_use,
    episode_to_use,
    explicit_root: Optional[str] = None,
) -> str:
    return str(
        Path(channel_root(paths_cfg, explicit_root))
        / f"assoc_{association_to_use}"
        / f"ep_{episode_to_use}"
        / channel_npz_name(paths_cfg)
    )


def metrics_dir_path(workdir: str, paths_cfg: Any = None) -> str:
    subdir = runtime_metrics_subdir(paths_cfg)
    if not subdir:
        return str(workdir)
    return str(Path(workdir) / subdir)


def ray_results_path(workdir: str, paths_cfg: Any = None) -> str:
    return str(Path(workdir) / ray_results_dirname(paths_cfg))
