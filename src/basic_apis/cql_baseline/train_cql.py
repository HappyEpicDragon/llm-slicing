"""
CQL 训练脚本（基于 d3rlpy）。

用法：
    python src/basic_apis/cql_baseline/train_cql.py \
        --dataset data/channel_generality/cql/dataset_expert.h5 \
        --output data/channel_generality/cql/models/expert_seed0 \
        --seed 0 \
        --n_steps 100000

    # 5 seeds 循环
    for seed in 0 1 2 3 4; do
        python src/basic_apis/cql_baseline/train_cql.py \
            --dataset data/channel_generality/cql/dataset_expert.h5 \
            --output data/channel_generality/cql/models/expert_seed${seed} \
            --seed $seed
    done
"""
import os
import argparse
import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_cql(dataset_path: str, output_dir: str, seed: int = 0, n_steps: int = 100000,
              device: str = "cuda:0"):
    try:
        import d3rlpy
    except ImportError:
        raise ImportError("请先安装 d3rlpy：pip install d3rlpy")

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "rb") as f:
        dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())
    obs_shape = dataset.episodes[0].observations.shape[1:]
    print(f"Dataset: {len(dataset.episodes)} episodes, obs_shape={obs_shape}")

    # 按文献 [37] 设置超参数（d3rlpy 2.x: initial_alpha / conservative_weight 替代 alpha）
    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=5e-4,
        critic_learning_rate=1e-2,
        initial_alpha=25.0,
        conservative_weight=25.0,  # CQL conservatism coefficient
        batch_size=256,
    ).create(device=device if torch.cuda.is_available() else "cpu")

    print(f"Training CQL (seed={seed}, n_steps={n_steps}) ...")
    cql.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=1000,
        save_interval=10,
        experiment_name=f"cql_seed{seed}",
        logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir=output_dir),
    )

    model_path = os.path.join(output_dir, "model.pt")
    try:
        cql.save(model_path)
    except Exception:
        cql.save_model(model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Train CQL baseline")
    parser.add_argument('--dataset', required=True, help='d3rlpy .h5 数据集路径')
    parser.add_argument('--output', required=True, help='模型输出目录')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_steps', type=int, default=100000)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    train_cql(args.dataset, args.output, args.seed, args.n_steps, args.device)


if __name__ == '__main__':
    main()
