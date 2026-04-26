"""PPO teacher training and evaluation handlers."""


def train_ppo_ha_weighted(sim):
    from src.basic_apis.ppo.ppo_ha_weighted.train import train

    train(
        sim.cfg.train_ppo_ha_weighted,
        paths_cfg=sim.paths_cfg,
        workdir=sim.workdir,
    )


def test_ppo_ha_weighted(sim):
    from src.basic_apis.ppo.ppo_ha_weighted.test import test_ppo_ha

    test_ppo_ha(
        sim.cfg.test_ppo_ha_weighted,
        paths_cfg=sim.paths_cfg,
        workdir=sim.workdir,
    )


def train_dt_teacher(sim):
    from src.basic_apis.dt.train_teacher import train

    train(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def test_dt_teacher(sim):
    from src.basic_apis.dt.test_teacher import test_dt_teacher

    test_dt_teacher(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)
