from omegaconf import OmegaConf
from .env_ray import env_creator
from .ray_agent import RayAgent
from src.basic_apis.asset_utils import build_versioned_run_dir, update_latest_symlink, ensure_clean_dir, ensure_dir


def train_ppo(cfg, paths_cfg=None, workdir=None):
    asset_cfg = cfg.get("asset", None)
    paths_cfg = paths_cfg if paths_cfg is not None else cfg.get("paths", None)
    workdir = workdir if workdir is not None else str(cfg.get("workdir", cfg.get("hydra_workdir", ".")))
    run_dir = None
    model_root = None
    use_asset = False
    if asset_cfg and bool(asset_cfg.get("use_versioned_runs", False)):
        skip_when_finetune = bool(asset_cfg.get("skip_when_finetune", True))
        if cfg.get("enable_finetune", False) and skip_when_finetune:
            use_asset = False
        else:
            use_asset = True
    if use_asset:
        model_root = str(asset_cfg.model_root)
        run_id_cfg = asset_cfg.get("run_id", "auto")
        run_id = None if str(run_id_cfg) == "auto" else str(run_id_cfg)
        run_dir = build_versioned_run_dir(model_root, run_id=run_id)
        if bool(asset_cfg.get("clean_before_run", False)):
            ensure_clean_dir(run_dir)
        else:
            ensure_dir(run_dir)
        cfg.hydra_workdir = run_dir
        workdir = run_dir
        print(f"[Asset] versioned run dir: {run_dir}")

    env_config = OmegaConf.to_container(cfg.environment, resolve=True)
    env_config["paths_cfg"] = paths_cfg
    env_config["workdir"] = workdir
    env_config['mode'] = cfg.env_updates.mode
    scenario_mode = cfg.env_updates.scenario_mode
    env_config['scenario_mode'] = scenario_mode
    env_config['model_name'] = cfg.env_updates.model_name
    env_config[scenario_mode]['training']['active_scenario_list'] = cfg.env_updates[scenario_mode].training.active_scenario_list
    env_config[scenario_mode]['testing']['active_scenario_list'] = cfg.env_updates[scenario_mode].testing.active_scenario_list
    env_config[scenario_mode]['evaluating']['active_scenario_list'] = cfg.env_updates[scenario_mode].evaluating.active_scenario_list
    ray_config = OmegaConf.to_container(cfg.agent, resolve=True)

    if cfg.get("enable_finetune", False):
        # 这里硬编码或从 cfg 读取 S2 Expert 的 Checkpoint 路径
        # 请替换为你实际硬盘上的 S2 Expert 绝对路径
        base_checkpoint_path = cfg.base_checkpoint_path

        env_config["finetune_checkpoint_path"] = base_checkpoint_path
        print(f"🎯 微调模式开启: 将加载 {base_checkpoint_path}")
    else:
        env_config["finetune_checkpoint_path"] = None

    agent = RayAgent(
        env_creator=env_creator,
        env_config=env_config,
        paths_cfg=paths_cfg,
        workdir=workdir,
        **ray_config
    )
    agent.train()
    if use_asset and bool(asset_cfg.get("update_latest", True)):
        if model_root is None:
            model_root = str(asset_cfg.model_root)
        if run_dir is None:
            run_dir = str(cfg.hydra_workdir)
        link_path = update_latest_symlink(model_root, run_dir)
        print(f"[Asset] updated latest symlink: {link_path}")