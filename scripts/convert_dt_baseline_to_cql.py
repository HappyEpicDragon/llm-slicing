"""
Convert ppo-baseline trajectories (dt_baseline_v2 format) to d3rlpy MDPDataset (.h5).

- obs:     145-dim flat observation (already normalized to [-1,1])
- actions: first 5-dim continuous PRB allocation (tanh, range [-1,1])
           discrete intra-scheduler dims (last 5) are discarded since
           CQL natively handles continuous action spaces.

Usage:
    pixi run python scripts/convert_dt_baseline_to_cql.py \
        --input_dir data/channel_generality/dt_baseline_v2/dataset/training/ \
        --output_path data/channel_generality/cql_baseline/dataset.h5
"""
import argparse
import glob
import os
import pickle

import numpy as np


def convert(input_dir: str, output_path: str):
    files = sorted(glob.glob(os.path.join(input_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No pkl files found in {input_dir}")
    print(f"Converting {len(files)} trajectories from {input_dir}")

    all_obs, all_actions, all_rewards, all_terminals = [], [], [], []

    for fpath in files:
        with open(fpath, "rb") as f:
            traj = pickle.load(f)

        obs = traj["observations"].astype(np.float32)    # (T, 145)
        act_full = traj["actions"].astype(np.float32)    # (T, 10)
        act_cont = act_full[:, :5]                        # (T, 5) continuous PRB
        rew = traj["rewards"].astype(np.float32)          # (T,)
        done = traj["dones"].astype(bool)                 # (T,)

        all_obs.append(obs)
        all_actions.append(act_cont)
        all_rewards.append(rew)
        all_terminals.append(done.astype(np.float32))

    observations = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    print(f"\nDataset: {len(observations)} transitions")
    print(f"  obs   shape={observations.shape}  range=[{observations.min():.3f}, {observations.max():.3f}]")
    print(f"  act   shape={actions.shape}  range=[{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  rew   range=[{rewards.min():.3f}, {rewards.max():.3f}]  mean={rewards.mean():.3f}")
    print(f"  terminal ratio: {terminals.mean():.4f}")

    try:
        import d3rlpy
    except ImportError:
        raise ImportError("d3rlpy not installed. Run: pixi add d3rlpy")

    dataset = d3rlpy.dataset.MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w+b") as f:
        dataset.dump(f)
    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    convert(args.input_dir, args.output_path)


if __name__ == "__main__":
    main()
