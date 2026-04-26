"""Decision Transformer transfer and dataset handlers."""


def collect_dt_data(sim):
    from src.basic_apis.dt.collect_data import collect

    collect(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def train_dt(sim):
    from src.basic_apis.dt.train import train_hydra

    train_hydra(sim.cfg)


def test_dt(sim):
    from src.basic_apis.dt.test import test_dt

    test_dt(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def train_dt_tiny(sim):
    from src.basic_apis.dt.train import train_hydra_tiny

    train_hydra_tiny(sim.cfg)


def test_dt_tiny(sim):
    from src.basic_apis.dt.test import test_dt_tiny

    test_dt_tiny(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)
