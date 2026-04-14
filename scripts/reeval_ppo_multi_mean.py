"""直接调用 test_ppo_baseline_multi，用 mean 口径重测 ppo_multi s5-s9。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from omegaconf import OmegaConf
from src.basic_apis.ppo.ppo_baseline.test_ppo_baseline import test_ppo_baseline_multi
from src.basic_apis.network_slicing_business.path_manager import PathManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pm = PathManager(ROOT)

env_cfg   = OmegaConf.load(os.path.join(ROOT, "conf/environment/single_scenario_env.yaml"))
agent_cfg = OmegaConf.load(os.path.join(ROOT, "conf/agent/ray_agent.yaml"))
sim_cfg   = OmegaConf.load(os.path.join(ROOT, "conf/simulation/channel_generality/test_ppo_baseline_multi.yaml"))

cfg = OmegaConf.merge(
    {"environment": env_cfg, "agent": agent_cfg},
    sim_cfg,
)
OmegaConf.set_struct(cfg, False)
cfg.test_ppo_baseline_multi.test_seeds = [0]
cfg.test_ppo_baseline_multi.save_root = "data/channel_generality/ppo_multi_mean"
cfg.test_ppo_baseline_multi.clean_before_save = True
cfg.test_ppo_baseline_multi.hydra_workdir = "outputs/ppo_multi_mean_eval"
os.makedirs("outputs/ppo_multi_mean_eval", exist_ok=True)

print("Re-evaluating ppo_multi on s5-s9 with mean drift aggregation...")
test_ppo_baseline_multi(cfg, pm)
print("Done.")
