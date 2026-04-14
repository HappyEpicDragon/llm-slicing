import pkgutil
import importlib

# 遍历包中的所有模块，动态导入它们
# 这会触发每个模块顶层的代码，包括我们的 @SimulationFactory.register 装饰器
for _, name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f".{name}", __package__)

# 导出核心类，方便外部如 main.py 使用
from .base_simu.base_simulation import BaseSimulation
from .base_simu.simulation_factory import SimulationFactory
