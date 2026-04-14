"""
Train DT-baseline on ppo-baseline trajectories (mixed continuous+discrete actions).

Uses flat 145-dim observations and model_baseline.py's DecisionTransformerBaseline.
Loss: MSE for 5 continuous inter-slice actions + CrossEntropy for 5 discrete intra-slice actions.
"""
import os, sys, argparse, json
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from src.basic_apis.dt_v2.model_baseline import build_dt_baseline

OBS_DIM = 145
CONT_DIM = 5
DISC_DIMS = [3, 3, 3, 3, 3]
ACT_DIM = CONT_DIM + len(DISC_DIMS)  # 10


class BaselineDTDataset(Dataset):

    def __init__(self, dataset_path, context_len=20, rtg_scale=1.0, metadata_path="metadata.json"):
        self.context_len = context_len
        self.rtg_scale = rtg_scale

        self.normalize = False
        if not os.path.exists(metadata_path):
            potential_path = os.path.join(os.path.dirname(dataset_path), "metadata.json")
            if os.path.exists(potential_path):
                metadata_path = potential_path

        if os.path.exists(metadata_path):
            print(f"Loading metadata from {metadata_path}")
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            obs_stats = meta.get("obs_stats", meta)
            if "flat" in obs_stats:
                self.obs_mean = np.array(obs_stats["flat"]["mean"], dtype=np.float32)
                self.obs_std = np.array(obs_stats["flat"]["std"], dtype=np.float32) + 1e-6
                self.normalize = True
            elif "mean" in obs_stats:
                self.obs_mean = np.array(obs_stats["mean"], dtype=np.float32)
                self.obs_std = np.array(obs_stats["std"], dtype=np.float32) + 1e-6
                self.normalize = True
        if not self.normalize:
            print("Metadata not found or no flat stats. Skipping normalization.")

        self.trajectories = []
        self.indices = []

        if os.path.isdir(dataset_path):
            files = sorted(f for f in os.listdir(dataset_path) if f.endswith('.pkl'))
            print(f"Loading {len(files)} trajectory files from {dataset_path}")
            for f_name in tqdm(files, desc="Loading"):
                try:
                    with open(os.path.join(dataset_path, f_name), 'rb') as f:
                        traj = pickle.load(f)
                        self._process(traj)
                except Exception as e:
                    print(f"Error loading {f_name}: {e}")
        else:
            with open(dataset_path, 'rb') as f:
                raw = pickle.load(f)
                trajs = raw if isinstance(raw, list) else [raw]
                for traj in trajs:
                    self._process(traj)

        print(f"Dataset loaded: {len(self.trajectories)} trajectories, {len(self.indices)} samples")

    def _process(self, traj):
        obs = np.asarray(traj['observations'], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        actions = np.asarray(traj['actions'], dtype=np.float32)
        rewards = np.nan_to_num(np.asarray(traj['rewards'], dtype=np.float32),
                                nan=0.0, posinf=0.0, neginf=-100.0)

        raw_rtg = self._discount_cumsum(rewards, gamma=1.0)
        rtg_transformed = np.sign(raw_rtg) * np.log1p(np.abs(raw_rtg))

        T = len(actions)
        entry = {
            'obs': obs[:T],
            'actions': actions[:T],
            'rtg': rtg_transformed[:T],
            'length': T,
        }
        self.trajectories.append(entry)
        for step in range(T):
            self.indices.append((len(self.trajectories) - 1, step))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        traj_idx, end_idx = self.indices[idx]
        traj = self.trajectories[traj_idx]

        start_idx = end_idx - self.context_len + 1
        if start_idx < 0:
            window_start = 0
            pad_len = abs(start_idx)
        else:
            window_start = start_idx
            pad_len = 0
        window_end = end_idx + 1

        obs = traj['obs'][window_start:window_end].copy()
        if self.normalize:
            safe_std = self.obs_std.copy()
            safe_std[safe_std < 1e-2] = 1.0
            obs = np.clip((obs - self.obs_mean) / safe_std, -5.0, 5.0)

        rtg = traj['rtg'][window_start:window_end]
        actions_target = traj['actions'][window_start:window_end]
        timesteps = np.arange(window_start, window_end)

        if window_start > 0:
            prev_act = traj['actions'][window_start - 1].reshape(1, -1)
            actions_input = np.concatenate([prev_act, actions_target[:-1]], axis=0)
        else:
            dummy = np.zeros((1, ACT_DIM), dtype=np.float32)
            actions_input = np.concatenate([dummy, actions_target[:-1]], axis=0) if len(actions_target) > 0 else dummy

        def pad(data, p, pad_val=0):
            t = torch.from_numpy(data).float()
            if p == 0:
                return t
            shape = t.shape
            return torch.cat([torch.full((p, *shape[1:]), pad_val, dtype=t.dtype), t], dim=0)

        K = window_end - window_start
        return {
            "obs": pad(obs, pad_len),
            "actions": pad(actions_input, pad_len),
            "actions_target": pad(actions_target, pad_len),
            "rtg": pad(rtg.reshape(-1, 1), pad_len),
            "timesteps": pad(timesteps.astype(np.float32), pad_len).long(),
            "mask": torch.cat([torch.zeros(pad_len), torch.ones(K)]),
        }

    @staticmethod
    def _discount_cumsum(x, gamma):
        out = np.zeros_like(x)
        out[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            out[t] = x[t] + gamma * out[t + 1]
        return out


def train(
    dataset_dir="data/channel_generality/dt_baseline/dataset",
    save_dir="data/channel_generality/dt_baseline/dt_model",
    epochs=30,
    batch_size=1024,
    lr=1e-4,
    context_len=20,
    embed_dim=512,
    n_layer=12,
    n_head=16,
    rtg_scale=100000.0,
    device_str="cuda",
    grad_accum=1,
    resume_from=None,
    total_epochs=None,
):
    device = torch.device(device_str)
    torch.set_num_threads(2)

    train_path = os.path.join(dataset_dir, "training")
    meta_path = os.path.join(dataset_dir, "metadata.json")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading dataset from {train_path}")
    train_dataset = BaselineDTDataset(
        train_path, context_len=context_len, rtg_scale=rtg_scale, metadata_path=meta_path,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )

    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create({
        "context_len": context_len, "embed_dim": embed_dim,
        "n_layer": n_layer, "n_head": n_head,
        "activation": "relu", "dropout": 0.1,
    })

    model = build_dt_baseline(
        model_cfg=model_cfg, obs_dim=OBS_DIM, cont_dim=CONT_DIM,
        disc_dims=DISC_DIMS, embed_dim=embed_dim,
    ).to(device)

    _specific_gpu = ":" in device_str
    if not _specific_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    start_epoch = 0
    if resume_from is not None:
        resume_path = str(resume_from)
        print(f"Resuming from checkpoint: {resume_path}")
        raw = torch.load(resume_path, map_location=device)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(raw)
        stem = Path(resume_path).stem
        if stem.startswith("epoch_"):
            start_epoch = int(stem.split("_")[1])
        print(f"  Loaded weights; resuming from epoch {start_epoch}")

    _total = total_epochs if total_epochs is not None else (start_epoch + epochs)
    _remaining = _total - start_epoch

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params, obs_dim={OBS_DIM}, "
          f"cont_dim={CONT_DIM}, disc_dims={DISC_DIMS}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_total)
    for _ in range(start_epoch):
        scheduler.step()

    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss(ignore_index=-100)

    scaler = torch.cuda.amp.GradScaler(enabled="cuda" in device_str)

    print(f"Training epochs {start_epoch+1}-{start_epoch+_remaining} "
          f"(total budget {_total}), batch_size={batch_size}, grad_accum={grad_accum}")

    for epoch in range(start_epoch, start_epoch + _remaining):
        model.train()
        losses_mse, losses_ce, losses_total = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+_remaining}")
        for step, batch in enumerate(pbar):
            obs = batch["obs"].to(device)
            actions_in = batch["actions"].to(device)
            actions_tgt = batch["actions_target"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled="cuda" in device_str):
                out = model(
                    states=obs, actions=actions_in, returns=rtg,
                    timesteps=timesteps, attention_mask=mask,
                )

                cont_preds = out["cont_preds"]  # (B, K, 5)
                disc_preds = out["disc_preds"]  # (B, K, 15)

                cont_tgt = actions_tgt[:, :, :CONT_DIM]
                disc_tgt = actions_tgt[:, :, CONT_DIM:].long()

                valid = mask.unsqueeze(-1).bool()

                mse_loss = mse_fn(
                    cont_preds[valid.expand_as(cont_preds)],
                    cont_tgt[valid.expand_as(cont_tgt)],
                )

                ce_loss = torch.tensor(0.0, device=device)
                disc_offset = 0
                for i, ds in enumerate(DISC_DIMS):
                    logits = disc_preds[:, :, disc_offset:disc_offset + ds]
                    targets = disc_tgt[:, :, i]
                    targets_masked = targets.clone()
                    targets_masked[~mask.bool()] = -100
                    ce_loss = ce_loss + ce_fn(
                        logits.reshape(-1, ds), targets_masked.reshape(-1),
                    )
                    disc_offset += ds

                loss = (mse_loss + ce_loss) / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss = loss.item() * grad_accum
            losses_mse.append(mse_loss.item())
            losses_ce.append(ce_loss.item())
            losses_total.append(total_loss)
            pbar.set_postfix({"loss": f"{total_loss:.3e}", "mse": f"{mse_loss.item():.3e}",
                              "ce": f"{ce_loss.item():.3e}"})

        scheduler.step()
        avg_total = np.mean(losses_total)
        avg_mse = np.mean(losses_mse)
        avg_ce = np.mean(losses_ce)
        print(f"  Epoch {epoch+1}: total={avg_total:.4e}  mse={avg_mse:.4e}  ce={avg_ce:.4e}")

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            ckpt = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, ckpt)
            print(f"  Saved {ckpt}")

    final = os.path.join(save_dir, "final_dt_baseline.pth")
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, final)
    print(f"Training done. Final model: {final}")


def train_hydra(cfg, path_manager):
    """Hydra entry point: called by channel_generality.py -> train_dt_baseline mode."""
    tc = cfg.train_dt_baseline
    resume_from = str(tc.resume_from) if getattr(tc, "resume_from", None) else None
    total_epochs = int(tc.optimizer.total_epochs) if getattr(tc.optimizer, "total_epochs", None) else None
    train(
        dataset_dir=str(tc.dataset_dir),
        save_dir=str(tc.save_dir),
        epochs=int(tc.optimizer.epochs),
        batch_size=int(tc.optimizer.batch_size),
        lr=float(tc.optimizer.lr),
        context_len=int(tc.model.context_len),
        embed_dim=int(tc.model.embed_dim),
        n_layer=int(tc.model.n_layer),
        n_head=int(tc.model.n_head),
        rtg_scale=float(tc.optimizer.rtg_scale),
        device_str=str(tc.device),
        grad_accum=int(tc.grad_accum),
        resume_from=resume_from,
        total_epochs=total_epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/channel_generality/dt_baseline/dataset")
    parser.add_argument("--save_dir", type=str, default="data/channel_generality/dt_baseline/dt_model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--total_epochs", type=int, default=None,
                        help="Total epoch budget for scheduler (resumed + new)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--context_len", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset, save_dir=args.save_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, context_len=args.context_len,
        embed_dim=args.embed_dim, n_layer=args.n_layer, n_head=args.n_head,
        grad_accum=args.grad_accum, device_str=args.device,
        resume_from=args.resume_from, total_epochs=args.total_epochs,
    )
