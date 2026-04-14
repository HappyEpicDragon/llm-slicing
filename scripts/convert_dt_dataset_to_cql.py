"""
Convert DT-v2 dataset (pkl files with structured obs + MultiDiscrete actions)
into CQL-compatible h5 format (flat obs + continuous 5-dim PRB proportion actions).

DT-v2 pkl:
  obs  = {inter_feat: (5,8), intra_feat: (25,7), global_feat: (2,)}
  act  = np.ndarray (T, 11)  [codebook_idx, 5× ordering, 5× intra]

CQL h5 (d3rlpy MDPDataset):
  obs  = (N, 45) float32     inter[:,:4].flat + intra[:,:5].reshape(5,5,5).mean(1).flat
  act  = (N, 5)  float32     per-slice PRB proportions (codebook + ordering → actual allocation)
  rew  = (N,)    float32
  term = (N,)    float32

Usage:
    pixi run python scripts/convert_dt_dataset_to_cql.py \
        --input_dir data/channel_generality/dt_v2_mean_rwd/dataset/training/ \
        --output_path data/channel_generality/cql_v2/dataset.h5
"""
import os
import argparse
import pickle
import numpy as np

from src.basic_apis.cql_baseline.cql_env_wrapper import INTER_QUOTA_PATTERNS

TOTAL_PRBS = 135
NUM_SLICES = 5
USERS_PER_SLICE = 5


def convert_obs_v2_to_flat45(obs_dict):
    """V2 structured obs → CQL-compatible flat 45-dim vector.

    Takes the V1-compatible feature subset (first 4 inter dims, first 5 intra dims)
    so the output matches CQLObsWrapper.transform_obs exactly.
    """
    inter = np.asarray(obs_dict["inter_feat"], dtype=np.float32)  # (5, 8)
    intra = np.asarray(obs_dict["intra_feat"], dtype=np.float32)  # (25, 7)

    inter_v1 = inter[:, :4]  # prio, drift, traffic, alloc_ratio
    intra_v1 = intra[:, :5]  # buffer, prio, drift_min, csi, hol

    intra_per_slice = intra_v1.reshape(NUM_SLICES, USERS_PER_SLICE, 5).mean(axis=1)
    return np.concatenate([inter_v1.flatten(), intra_per_slice.flatten()])  # (45,)


def convert_action_v2(action, codebook):
    """V2 MultiDiscrete action → per-slice PRB fraction (5-dim continuous).

    Reconstructs the actual per-slice allocation by applying the ordering action
    to the codebook template, mirroring HierarchicalSlicingEnvV2.step().
    """
    codebook_idx = int(action[0])
    order_actions = action[1 : 1 + NUM_SLICES]

    raw_pattern = codebook[codebook_idx]
    sorted_quotas = np.sort(raw_pattern)[::-1]
    sorted_slice_indices = np.argsort(order_actions)[::-1]

    inter_quotas = np.zeros(NUM_SLICES, dtype=np.float32)
    for rank, s_idx in enumerate(sorted_slice_indices):
        inter_quotas[s_idx] = sorted_quotas[rank]

    return inter_quotas / float(TOTAL_PRBS)


def main():
    parser = argparse.ArgumentParser(
        description="Convert DT-v2 dataset to CQL-compatible h5 format"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Dir containing DT-v2 pkl files (e.g. .../dataset/training/)",
    )
    parser.add_argument("--output_path", required=True, help="Output .h5 file path")
    parser.add_argument(
        "--max_trajs", type=int, default=None,
        help="Limit the number of trajectories to convert",
    )
    args = parser.parse_args()

    try:
        import d3rlpy
    except ImportError:
        raise ImportError("d3rlpy is required: pip install d3rlpy")

    codebook = INTER_QUOTA_PATTERNS

    pkl_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {args.input_dir}")
    if args.max_trajs is not None:
        pkl_files = pkl_files[: args.max_trajs]

    print(f"Converting {len(pkl_files)} trajectories from {args.input_dir}")
    print(f"Codebook size: {len(codebook)}, total PRBs: {TOTAL_PRBS}")

    observations, actions, rewards, terminals = [], [], [], []

    for fname in pkl_files:
        with open(os.path.join(args.input_dir, fname), "rb") as f:
            traj = pickle.load(f)

        traj_obs = traj["observations"]  # list[dict]
        traj_acts = traj["actions"]  # (T, 11) ndarray
        traj_rews = traj["rewards"]  # (T,) ndarray
        traj_dones = traj["dones"]  # (T,) bool ndarray

        T = min(len(traj_obs), len(traj_acts), len(traj_rews))
        if T == 0:
            continue

        for t in range(T):
            observations.append(convert_obs_v2_to_flat45(traj_obs[t]))
            actions.append(convert_action_v2(traj_acts[t], codebook))
            rewards.append(float(traj_rews[t]))

            done = bool(traj_dones[t]) if t < len(traj_dones) else (t == T - 1)
            terminals.append(done)

    if not observations:
        raise ValueError("Empty dataset after conversion — check pkl contents")

    obs_arr = np.array(observations, dtype=np.float32)
    act_arr = np.array(actions, dtype=np.float32)
    rew_arr = np.array(rewards, dtype=np.float32)
    ter_arr = np.array(terminals, dtype=np.float32)

    print(f"\nDataset: {len(obs_arr)} transitions")
    print(f"  obs  shape={obs_arr.shape}  range=[{obs_arr.min():.3f}, {obs_arr.max():.3f}]")
    print(f"  act  shape={act_arr.shape}  range=[{act_arr.min():.3f}, {act_arr.max():.3f}]")
    print(f"  rew  range=[{rew_arr.min():.3f}, {rew_arr.max():.3f}]  mean={rew_arr.mean():.3f}")
    print(f"  terminal ratio: {ter_arr.mean():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    dataset = d3rlpy.dataset.MDPDataset(
        observations=obs_arr,
        actions=act_arr,
        rewards=rew_arr,
        terminals=ter_arr,
    )
    with open(args.output_path, "w+b") as fh:
        dataset.dump(fh)
    print(f"\nSaved to: {args.output_path}")


if __name__ == "__main__":
    main()
