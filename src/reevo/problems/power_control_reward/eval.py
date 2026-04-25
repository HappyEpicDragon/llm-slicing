"""
Evaluation script for Power Control reward function evolution.

Usage (called by ReEvo framework):
    python -u eval.py {problem_size} {root_dir} train

Args:
    problem_size: K (number of TX-RX pairs)
    root_dir: ReEvo project root (for locating gpt.py)
    mood: "train" (search phase) or "val" (final validation)

Environment variables (injected by EvoSimulator):
    PC_PROJECT_ROOT: llm_slicing project root
    PC_PROXY_TRAIN_RATIO: PPO training timesteps ratio (float, default 0.5)
    PC_SEED: random seed (int, default 42)

Output:
    Prints `##FITNESS## <scalar>` (lower is better); ReEvo parses from stdout.
"""
import sys
import os
import math
import importlib
import traceback

# Parse command line arguments
problem_size = sys.argv[1]  # K value
root_dir = sys.argv[2]  # ReEvo root (gpt.py parent directory)
mood = sys.argv[3] if len(sys.argv) > 3 else "train"

# Read environment variables
project_root = os.environ.get("PC_PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
proxy_train_ratio = float(os.environ.get("PC_PROXY_TRAIN_RATIO", "0.5"))
seed = int(os.environ.get("PC_SEED", "42"))
K = int(problem_size)

# Configure import paths
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

# Import LLM-generated reward function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
gpt_module_name = os.environ.get("REEVO_GPT_MODULE", "gpt")
gpt = importlib.import_module(gpt_module_name)
if not hasattr(gpt, 'compute_reward'):
    gpt.compute_reward = next(getattr(gpt, n) for n in dir(gpt) if n.startswith('compute_reward_v'))

print(f"[eval.py] started: K={K} ratio={proxy_train_ratio} seed={seed}", flush=True)


# Safety check: validate reward function
def safety_check(reward_fn):
    """Verify generated reward function is valid."""
    import numpy as np
    import random
    test_results = []
    try:
        for trial in range(24):
            k = random.randint(4, 16)
            rates = np.random.rand(k) * 5.0
            qos_viol = random.uniform(0.0, 1.0)
            fairness = random.uniform(0.0, 1.0)

            result = reward_fn(rates, qos_viol, fairness)
            if not isinstance(result, (int, float, np.number)):
                return False, f"Output is not a scalar: {type(result)}"
            if not math.isfinite(result):
                return False, f"Output is not finite: {result}"
            test_results.append(float(result))
    except Exception as e:
        return False, f"Runtime error: {e}\n{traceback.format_exc()}"

    if len(set(round(r, 6) for r in test_results)) <= 1:
        return False, "Constant function (all outputs identical)"
    return True, "OK"


# Execute safety check
passed, msg = safety_check(gpt.compute_reward)
if not passed:
    print(f"[eval.py] Safety check failed: {msg}", file=sys.stderr)
    print(f"##FITNESS## {math.inf}")
    sys.exit(0)


# Train PPO and evaluate
def train_and_evaluate():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()

    # Load configuration
    config_dir = os.path.join(project_root, "conf")
    initialize_config_dir(config_dir=config_dir, version_base="1.1")

    cfg = compose(config_name="conf", overrides=[
        "simulation=power_control/train_ppo_discrete",
        f"environment.K={K}",
        f"environment.N={K}",
    ])

    # Adjust training timesteps based on proxy ratio
    original_timesteps = cfg.training.total_timesteps
    adjusted_timesteps = int(original_timesteps * proxy_train_ratio)

    print(f"[eval.py] Training timesteps: {adjusted_timesteps}")

    # Import train function
    from src.basic_apis.power_control.train import train

    # Wrap reward function for environment
    def reward_wrapper(rates, qos_violations, jains_fairness):
        return gpt.compute_reward(rates, qos_violations, jains_fairness)

    # Train PPO with injected reward function
    print("[eval.py] Starting PPO training...")
    try:
        metrics = train(
            env_config=dict(cfg.environment),
            ppo_config=dict(cfg.ppo),
            total_timesteps=adjusted_timesteps,
            save_path=None,
            reward_fn=reward_wrapper,
            seed=seed,
            verbose=1
        )
    except Exception as e:
        print(f"[eval.py] Training failed: {e}", file=sys.stderr)
        traceback.print_exc()
        print(f"##FITNESS## {math.inf}")
        sys.exit(0)

    return metrics


# Execute training
metrics = train_and_evaluate()

# Compute fitness (use eval metrics for final fitness)
qos_viol = metrics.get("eval_avg_qos_violation", 1.0)
fairness = metrics.get("eval_avg_fairness", 0.0)
sum_rate = metrics.get("eval_avg_sum_rate", 0.0)

fairness_threshold = 0.5 + 0.5 / K
fairness_gap = max(0, fairness_threshold - fairness)
fitness = 10.0 * qos_viol + fairness_gap

# Output results
print(f"\n[RESULTS]")
print(f"  Sum Rate: {sum_rate:.2f} bps/Hz")
print(f"  QoS Violation Rate: {qos_viol:.4f}")
print(f"  Jain's Fairness: {fairness:.4f}")
print(f"  Fitness: {fitness:.4f}")
print(f"\n##FITNESS## {fitness}")
