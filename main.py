import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import sys

try:
    from src.simulation import SimulationFactory, BaseSimulation
except ImportError as e:
    print("错误：无法导入 'simulations' 模块。")
    print(f"原始错误: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="conf")
def main(cfg: DictConfig) -> None:
    try:
        log.info("=================================================")
        log.info("🚀 开始执行仿真项目")
        log.info("=================================================")
        # log.info("Hydra 加载的配置:\n" + OmegaConf.to_yaml(cfg))

        # 直接从配置中读取 name，不再需要复杂的解析
        sim_type = cfg.name
        log.info(f"🎯 Current Experiment Name: '{sim_type}'")
        log.info(f"🎯 Current Experiment Step: '{cfg.mode}'")

        available_sims = SimulationFactory.list_available()
        # log.info(f"📋 当前已注册的可用仿真: {available_sims}")

        # log.info("🏭 工厂正在创建仿真实例...")
        simulator: BaseSimulation = SimulationFactory.create(sim_type, cfg)
        # log.info(f"✅ 成功创建实例: {simulator.__class__.__name__}")

        log.info("🏃‍ 开始运行仿真...")
        simulator.run()
        log.info("🎉 仿真成功完成！")
        log.info("=================================================")

    except ValueError as e:
        log.error(f"❌ 配置或创建错误: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        log.error(f"❌ 仿真运行时发生未处理的异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()