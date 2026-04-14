"""
将 IDT pkl 格式数据集转换为 d3rlpy MDPDataset 格式。

用法：
    python src/basic_apis/cql_baseline/convert_dataset.py \
        --input data/channel_generality/dt/dataset/training/ \
        --output data/channel_generality/cql/dataset_expert.h5 \
        --variant expert

    python src/basic_apis/cql_baseline/convert_dataset.py \
        --input data/channel_generality/dt/dataset_raw/training/ \
        --output data/channel_generality/cql/dataset_raw.h5 \
        --variant raw
"""
import os
import argparse
import pickle
import numpy as np

from src.basic_apis.cql_baseline.cql_env_wrapper import CQLObsWrapper, INTER_QUOTA_PATTERNS


def convert_to_d3rlpy_format(pkl_dir: str, output_path: str, variant: str = "expert"):
    """将 IDT pkl 轨迹数据转为 d3rlpy MDPDataset 并保存。"""
    try:
        import d3rlpy
    except ImportError:
        raise ImportError("请先安装 d3rlpy：pip install d3rlpy")

    wrapper = CQLObsWrapper()
    observations, actions, rewards, terminals = [], [], [], []

    pkl_files = sorted([f for f in os.listdir(pkl_dir) if f.endswith('.pkl')])
    if not pkl_files:
        raise FileNotFoundError(f"在 {pkl_dir} 中未找到任何 .pkl 文件")

    print(f"Processing {len(pkl_files)} pkl files from {pkl_dir} (variant={variant})")

    total_prbs = 135
    for fname in pkl_files:
        fpath = os.path.join(pkl_dir, fname)
        with open(fpath, 'rb') as f:
            traj = pickle.load(f)

        traj_obs = traj.get('observations', [])
        traj_acts = traj.get('actions', [])
        traj_rews = traj.get('rewards', [])
        traj_dones = traj.get('dones', [])

        T = min(len(traj_obs), len(traj_acts), len(traj_rews))
        if T == 0:
            continue

        for t in range(T):
            flat_obs = wrapper.transform_obs(traj_obs[t])
            observations.append(flat_obs)

            act = np.array(traj_acts[t], dtype=np.float32)
            if act.shape[0] == 10:
                # raw/dt_baseline 格式：act[:5] = inter 连续分数，act[5:] = intra 离散 ID
                # 用 softmax 归一化确保所有值为正且和为 1
                fracs = act[:5]
                fracs = fracs - fracs.max()
                fracs = np.exp(fracs)
                continuous_frac = fracs / fracs.sum()
            else:
                # expert/dt 格式：act[0] = template_id, act[1:6] = intra IDs
                template_id = int(act[0])
                pattern = INTER_QUOTA_PATTERNS[template_id]
                continuous_frac = pattern.astype(np.float32) / float(total_prbs)
            actions.append(continuous_frac)

            rewards.append(float(traj_rews[t]))
            done = bool(traj_dones[t]) if t < len(traj_dones) else (t == T - 1)
            terminals.append(done)

    if not observations:
        raise ValueError("转换后数据集为空，请检查 pkl 文件内容")

    obs_arr = np.array(observations, dtype=np.float32)
    act_arr = np.array(actions, dtype=np.float32)
    rew_arr = np.array(rewards, dtype=np.float32)
    ter_arr = np.array(terminals, dtype=np.float32)  # v2 requires float32

    print(f"Dataset stats: {len(obs_arr)} transitions, obs_dim={obs_arr.shape[1]}, act_dim={act_arr.shape[1]}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        terminals=ter_arr,
    )
    # h5py 写入时内部会读文件对象，必须用 w+b 否则会报 io.UnsupportedOperation: read
    with open(output_path, "w+b") as f:
        dataset.dump(f)
    print(f"Saved dataset to: {output_path}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Convert IDT dataset to d3rlpy format")
    parser.add_argument('--input', required=True, help='IDT pkl 数据集目录')
    parser.add_argument('--output', required=True, help='输出 .h5 文件路径')
    parser.add_argument('--variant', default='expert', choices=['expert', 'raw'],
                        help='数据集变体标识（仅用于日志）')
    args = parser.parse_args()
    convert_to_d3rlpy_format(args.input, args.output, args.variant)


if __name__ == '__main__':
    main()
