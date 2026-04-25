"""
eval.py for IDT reward function search

调用方式（由 ReEvo 框架自动执行）：
    python -u eval.py {problem_size} {root_dir} train

其中：
    problem_size : proxy_scenario（int as string），传给 PPO 训练
    root_dir     : ReEvo 项目根目录（用于定位 gpt.py）
    mood         : "train"（搜索阶段）或 "val"（最终验证，暂未使用）

环境变量（由 EvoSimulator 注入）：
    IDT_PROJECT_ROOT      : llm_slicing 项目根目录
    IDT_PROXY_TRAIN_RATIO : PPO 训练步数比例 (float, 默认 0.5)
    IDT_SEED              : 随机种子 (int, 默认 42)
    IDT_PROXY_SCENARIOS   : 多场景 proxy fitness 的场景列表 (逗号分隔整数, 如 "0,2")；为空时 fallback 到单场景

输出：
    打印一行 `##FITNESS## <scalar>`（越小越好）；ReEvo 用正则从 stdout 解析。
"""
import sys
import os
import math
import importlib
import traceback

# ---- 解析命令行参数 ----
problem_size = sys.argv[1]   # proxy_scenario
root_dir = sys.argv[2]       # ReEvo 根目录（gpt.py 所在目录的父目录）
mood = sys.argv[3] if len(sys.argv) > 3 else "train"

# ---- 读取环境变量 ----
project_root = os.environ.get("IDT_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
proxy_train_ratio = float(os.environ.get("IDT_PROXY_TRAIN_RATIO", "0.5"))
seed = int(os.environ.get("IDT_SEED", "42"))
proxy_scenarios_str = os.environ.get("IDT_PROXY_SCENARIOS", "")
proxy_scenario = int(problem_size)

# ---- 配置 import 路径 ----
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# ---- 导入 LLM 生成的 reward function ----
# ReEvo 将 gpt*.py 写入 problems/idt_reward/（默认 gpt.py，并发时为 gpt_<id>.py）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
gpt_module_name = os.environ.get("REEVO_GPT_MODULE", "gpt")
gpt = importlib.import_module(gpt_module_name)
# reevo 写入的函数名带版本号（如 compute_reward_v2），统一别名为 compute_reward
if not hasattr(gpt, 'compute_reward'):
    gpt.compute_reward = next(getattr(gpt, n) for n in dir(gpt) if n.startswith('compute_reward_v'))

# 早期输出：让 ReEvo 的 block_until_running 快速返回（不等到 PPO 打印）
print(f"[eval.py] started: scenario={proxy_scenario} ratio={proxy_train_ratio} seed={seed}", flush=True)


# ---- Safety Gate：静态检查 ----
def safety_check(reward_fn):
    """验证生成的 reward function 是否合法。"""
    import random
    test_results = []
    try:
        for trial in range(24):
            n = random.randint(2, 4)
            min_m = [random.uniform(-1.0, 1.0) for _ in range(n)]
            mean_m = [random.uniform(-1.0, 1.0) for _ in range(n)]
            metric_m = [{"thr": random.uniform(-1, 1), "rel": random.uniform(-1, 1), "lat": random.uniform(-1, 1)} for _ in range(n)]
            if trial < 2:
                is_hp = [True] * n       # edge: all HP
            elif trial < 4:
                is_hp = [False] * n      # edge: all NHP
            else:
                is_hp = [random.random() > 0.6 for _ in range(n)]
            prev_min = [random.uniform(-1, 1) for _ in range(n)]
            prev_mean = [random.uniform(-1, 1) for _ in range(n)]
            mean_buf = [random.uniform(0, 1) for _ in range(n)]
            max_buf = [random.uniform(0, 1) for _ in range(n)]

            result = reward_fn(min_m, mean_m, metric_m, is_hp, n, prev_min, prev_mean, mean_buf, max_buf)
            if not isinstance(result, (int, float)):
                return False, f"Output is not a scalar: {type(result)}"
            if not math.isfinite(result):
                return False, f"Output is not finite: {result}"
            # 注意：不做 range 检查。env 层会 clamp 到 [-10, 10]；
            # 函数本身产生 ±50 以外的值是合法的（如强烈的 HP 指数惩罚）。
            test_results.append(float(result))
    except Exception as e:
        return False, f"Runtime error: {e}\n{traceback.format_exc()}"

    if len(set(round(r, 6) for r in test_results)) <= 1:
        return False, "Constant function (all outputs identical)"
    return True, "OK"


# ---- 执行安全检查 ----
passed, msg = safety_check(gpt.compute_reward)
if not passed:
    print(f"[eval.py] Safety check failed: {msg}", file=sys.stderr)
    print(f"##FITNESS## {math.inf}")
    sys.exit(0)


# ---- 训练 PPO 并评估 ----
def train_and_evaluate(scenario=None):
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    if scenario is None:
        scenario = proxy_scenario

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="reevo_eval_")

    GlobalHydra.instance().clear()
    conf_dir = os.path.join(project_root, "conf")
    with initialize_config_dir(config_dir=conf_dir, job_name="evo_eval", version_base=None):
        cfg = compose(
            config_name="conf",
            overrides=[
                f"simulation=channel_generality/train_ppo_ha_weighted",
                f"train_scenario={scenario}",
                f"workdir={tmp_dir}",   # 覆盖 ${hydra:runtime.output_dir}，使插值链可解析
                f"train_ppo_ha_weighted.asset.model_root={tmp_dir}",
                "train_ppo_ha_weighted.asset.update_latest=false",
            ],
        )
        train_cfg = cfg.train_ppo_ha_weighted
        full_steps = int(train_cfg.environment.train_rl.total_timesteps)
        proxy_steps = max(1000, int(full_steps * proxy_train_ratio))
        # 在 with 块内解析，此时 OmegaConf 交叉引用和 workdir 均已具体化
        train_cfg_dict = OmegaConf.to_container(train_cfg, resolve=True)

    train_cfg_dict["environment"]["train_rl"]["total_timesteps"] = proxy_steps
    proxy_train_cfg = OmegaConf.create(train_cfg_dict)

    from src.basic_apis.network_slicing_business.path_context import create_path_context
    from src.basic_apis.ppo.ppo_ha_weighted.train import train
    path_context = create_path_context(
        cfg,
        hydra_workdir=tmp_dir,
        project_root=project_root,
    )

    # 训练 PPO（注入 reward fn）
    try:
        metrics = train(proxy_train_cfg, path_context, reward_fn=gpt.compute_reward, seed=seed)
        hp_viol = float(metrics.get("hp_violation_rate", 1.0))
        nhp_viol = float(metrics.get("nhp_violation_rate", 1.0))
    except Exception as e:
        print(f"[eval.py] PPO training failed: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"##FITNESS## {math.inf}")
        sys.exit(0)
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)

    fitness = 10.0 * hp_viol + nhp_viol
    return fitness


if mood == "val" and proxy_scenario < 0:
    # Case 1: validation 模式，遍历全部 5 个场景
    val_scenarios = list(range(5))
    print(f"[eval.py] Validation mode: evaluating on scenarios {val_scenarios}")
    fitnesses = []
    for sc in val_scenarios:
        f = train_and_evaluate(scenario=sc)
        print(f"[eval.py] scenario={sc} fitness={f:.6f}")
        fitnesses.append(f)
    fitness = sum(fitnesses) / len(fitnesses)
    print(f"[eval.py] mean_fitness={fitness:.6f}")
elif proxy_scenarios_str:
    # Case 2: 多场景 proxy fitness（搜索阶段，由 IDT_PROXY_SCENARIOS 控制）
    proxy_scenarios_list = [int(s.strip()) for s in proxy_scenarios_str.split(",") if s.strip()]
    print(f"[eval.py] Multi-scenario proxy: scenarios={proxy_scenarios_list}")
    fitnesses = []
    for sc in proxy_scenarios_list:
        f = train_and_evaluate(scenario=sc)
        print(f"[eval.py] scenario={sc} fitness={f:.6f}")
        fitnesses.append(f)
    fitness = sum(fitnesses) / len(fitnesses)
    print(f"[eval.py] mean_fitness={fitness:.6f}")
else:
    # Case 3: 单场景 fallback
    fitness = train_and_evaluate()
    print(f"[eval.py] fitness={fitness:.6f}")
print(f"##FITNESS## {fitness}")
