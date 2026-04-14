"""
推理时间基准测试脚本（E1 实验）。
测量各方法的单步推理延迟，验证 IDT 满足 < 1ms TTI 约束。

与 Tab.V 方法列表对齐：
  - PPO-MLP (Ray Agent)
  - CQL (d3rlpy model)
  - Lagrangian PPO (SB3 model)
  - PPO Discrete Teacher (SB3 model)
  - IDT (DecisionTransformer)

用法：
    python main.py name=channel_generality mode=inference_benchmark
"""
import os
import json
import time
import numpy as np
import torch
from types import SimpleNamespace
from omegaconf import DictConfig
from typing import Optional, Tuple


def benchmark_model(
    model_fn,
    sample_input,
    n_warmup: int = 100,
    n_runs: int = 1000,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """
    通用推理延迟测量函数。

    Args:
        model_fn:     无参可调用对象，执行一次推理
        sample_input: 输入样本（用于预热，具体使用由 model_fn 负责）
        n_warmup:     预热次数
        n_runs:       正式测量次数
        device:       推理设备

    Returns:
        (mean_ms, std_ms) — 单次推理延迟的均值和标准差（毫秒）
    """
    # 预热
    for _ in range(n_warmup):
        model_fn()

    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()

    times_ns = []
    for _ in range(n_runs):
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        model_fn()
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        times_ns.append(t1 - t0)

    times_ms = np.array(times_ns) / 1e6  # ns → ms
    return float(np.mean(times_ms)), float(np.std(times_ms))


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model_path: str) -> float:
    """获取模型文件大小（MB）"""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return float('nan')


def _benchmark_idt(cfg, device: str = 'cpu') -> dict:
    """测量 IDT 推理延迟"""
    from src.basic_apis.dt_utils.model_ha_dt import HierarchicalStateEncoder, build_decision_model
    from src.basic_apis.dt_utils.action_dims import resolve_action_dims_from_env_cfg
    from src.basic_apis.general_utils import pad_stack_tensor

    dt_cfg = cfg.get('dt_testing', cfg)
    model_path = str(dt_cfg.get('model_path', ''))
    if not os.path.exists(model_path):
        return {"method": "IDT", "status": "model_not_found", "model_path": model_path}

    model_cfg = dt_cfg.model if hasattr(dt_cfg, "model") else dt_cfg.get('model', {})
    if isinstance(model_cfg, dict):
        model_cfg = SimpleNamespace(**model_cfg)

    context_len = int(getattr(model_cfg, "context_len", 20))
    embed_dim = int(getattr(model_cfg, "embed_dim", 512))
    action_dims = resolve_action_dims_from_env_cfg(cfg.environment, intra_mode_count=3)

    state_encoder = HierarchicalStateEncoder(embed_dim=embed_dim)
    model = build_decision_model(
        model_cfg=model_cfg,
        state_encoder=state_encoder,
        action_dims=action_dims,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 构造 dummy 输入
    B, T = 1, context_len
    dummy_states = {
        'inter': torch.zeros(B, T, 5, 4).to(device),
        'intra': torch.zeros(B, T, 25, 5).to(device),
        'global': torch.zeros(B, T, 2).to(device),
    }
    dummy_actions = torch.zeros(B, T, len(action_dims), dtype=torch.long).to(device)
    dummy_rtg = torch.zeros(B, T, 1).to(device)
    dummy_steps = torch.zeros(B, T, dtype=torch.long).to(device)
    dummy_mask = torch.ones(B, T).to(device)

    def infer():
        with torch.no_grad():
            model(dummy_states, dummy_actions, dummy_rtg, dummy_steps, dummy_mask)

    mean_ms, std_ms = benchmark_model(infer, None, n_warmup=50, n_runs=500, device=device)
    n_params = count_parameters(model)
    size_mb = get_model_size_mb(model_path)

    return {
        "method": "IDT (Proposed)",
        "params": n_params,
        "size_mb": round(size_mb, 2),
        "inference_mean_ms": round(mean_ms, 4),
        "inference_std_ms": round(std_ms, 4),
        "meets_1ms_tti": mean_ms < 1.0,
    }


def _benchmark_sb3_ppo(model_path: str, obs_dim: int, method_name: str,
                       device: str = 'cpu') -> dict:
    """测量 SB3 PPO 推理延迟（适用于 PPO-HA-Weighted / Lagrangian PPO）"""
    try:
        from stable_baselines3 import PPO
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            return {"method": method_name, "status": "model_not_found"}
        model = PPO.load(model_path, device=device)
        dummy_obs = np.zeros(obs_dim, dtype=np.float32)

        def infer():
            model.predict(dummy_obs, deterministic=True)

        mean_ms, std_ms = benchmark_model(infer, None, n_warmup=100, n_runs=1000, device=device)
        policy = model.policy
        n_params = count_parameters(policy)
        size_mb = get_model_size_mb(model_path + ".zip")

        return {
            "method": method_name,
            "params": n_params,
            "size_mb": round(size_mb, 2),
            "inference_mean_ms": round(mean_ms, 4),
            "inference_std_ms": round(std_ms, 4),
            "meets_1ms_tti": mean_ms < 1.0,
        }
    except Exception as e:
        return {"method": method_name, "status": f"error: {e}"}


def _benchmark_cql(model_path: str, obs_dim: int = 45, device: str = 'cpu') -> dict:
    """测量 CQL 推理延迟（d3rlpy 模型）"""
    try:
        import d3rlpy
        if not os.path.exists(model_path):
            return {"method": "CQL [37]", "status": "model_not_found"}
        params_path = os.path.join(os.path.dirname(model_path), 'params.json')
        cql = d3rlpy.algos.CQL.from_json(params_path)
        cql.load_model(model_path)

        dummy_obs = np.zeros((1, obs_dim), dtype=np.float32)

        def infer():
            cql.predict(dummy_obs)

        mean_ms, std_ms = benchmark_model(infer, None, n_warmup=100, n_runs=1000, device=device)
        size_mb = get_model_size_mb(model_path)

        return {
            "method": "CQL [37]",
            "params": "N/A",
            "size_mb": round(size_mb, 2),
            "inference_mean_ms": round(mean_ms, 4),
            "inference_std_ms": round(std_ms, 4),
            "meets_1ms_tti": mean_ms < 1.0,
        }
    except Exception as e:
        return {"method": "CQL [37]", "status": f"error: {e}"}


def inference_benchmark(cfg: DictConfig, path_manager):
    """推理时间基准测试主入口（channel_generality.py 调用）"""
    bench_cfg = cfg.get('inference_benchmark', cfg)
    device = str(bench_cfg.get('device', 'cpu'))
    save_dir = str(bench_cfg.get('save_dir', 'data/channel_generality/inference_benchmark'))
    os.makedirs(save_dir, exist_ok=True)

    results = []
    print(f"\n{'='*60}")
    print(f"  Inference Latency Benchmark (device={device})")
    print(f"{'='*60}")

    # IDT
    idt_result = _benchmark_idt(cfg, device=device)
    results.append(idt_result)

    # PPO-HA-Weighted（SB3）
    ppo_ha_path = str(bench_cfg.get('ppo_ha_model_path', ''))
    if ppo_ha_path:
        results.append(_benchmark_sb3_ppo(ppo_ha_path, 45, 'PPO Discrete (Teacher)', device))

    # Lagrangian PPO（SB3）
    lag_path = str(bench_cfg.get('ppo_lagrangian_model_path', ''))
    if lag_path:
        results.append(_benchmark_sb3_ppo(lag_path, 45, 'Lagrangian PPO [20]', device))

    # CQL（d3rlpy）
    cql_path = str(bench_cfg.get('cql_model_path', ''))
    if cql_path:
        results.append(_benchmark_cql(cql_path, obs_dim=45, device=device))

    # 打印结果表
    print(f"\n{'Method':<28} {'Params':>10} {'Size(MB)':>10} {'Mean(ms)':>10} {'Std(ms)':>9} {'<1ms TTI':>10}")
    print("-" * 80)
    for r in results:
        if 'inference_mean_ms' in r:
            print(f"{r['method']:<28} "
                  f"{str(r.get('params','N/A')):>10} "
                  f"{str(r.get('size_mb','N/A')):>10} "
                  f"{r['inference_mean_ms']:>10.4f} "
                  f"{r['inference_std_ms']:>9.4f} "
                  f"{'✓' if r.get('meets_1ms_tti') else '✗':>10}")
        else:
            print(f"{r['method']:<28} {r.get('status','')}")

    # 保存 JSON
    save_path = os.path.join(save_dir, "inference_benchmark.json")
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {save_path}")

    return results
