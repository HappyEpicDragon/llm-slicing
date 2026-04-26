"""Baseline solver handlers kept for objective-synthesis comparisons."""


def train_ppo_baseline(sim):
    from src.basic_apis.ppo.ppo_baseline.train_ppo_baseline import train_ppo

    train_ppo(sim.cfg.train_ppo_baseline, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def test_ppo_baseline(sim):
    from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline

    test_ppo_baseline(sim.cfg.test_ppo_baseline, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def train_ppo_baseline_across_scenario(sim):
    from src.basic_apis.ppo.ppo_baseline.train_ppo_baseline import train_ppo

    train_ppo(
        sim.cfg.train_ppo_baseline_across_scenario,
        paths_cfg=sim.paths_cfg,
        workdir=sim.workdir,
    )


def test_ppo_baseline_finetune(sim):
    from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_finetune

    test_ppo_baseline_finetune(
        sim.cfg.test_ppo_baseline_finetune,
        paths_cfg=sim.paths_cfg,
        workdir=sim.workdir,
    )


def test_ppo_baseline_multi(sim):
    from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_multi

    test_ppo_baseline_multi(
        sim.cfg.test_ppo_baseline_multi,
        paths_cfg=sim.paths_cfg,
        workdir=sim.workdir,
    )


def train_dt_baseline(sim):
    from src.basic_apis.dt.train_baseline import train_hydra

    train_hydra(sim.cfg)


def test_dt_baseline(sim):
    from src.basic_apis.dt.test_baseline import test_dt_baseline

    test_dt_baseline(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def train_ppo_lagrangian(sim):
    from src.basic_apis.ppo.ppo_lagrangian.train import train_ppo_lagrangian

    train_ppo_lagrangian(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def test_ppo_lagrangian(sim):
    from src.basic_apis.ppo.ppo_lagrangian.test import test_ppo_lagrangian

    test_ppo_lagrangian(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def train_ppo_lagrangian_baseline(sim):
    from src.basic_apis.ppo.ppo_lagrangian.train_baseline import train_ppo_lagrangian_baseline

    train_ppo_lagrangian_baseline(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)


def test_ppo_lagrangian_baseline(sim):
    from src.basic_apis.ppo.ppo_lagrangian.test_baseline import test_ppo_lagrangian_baseline

    test_ppo_lagrangian_baseline(sim.cfg, paths_cfg=sim.paths_cfg, workdir=sim.workdir)
