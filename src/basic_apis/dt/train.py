"""
Phase 3b: Train DT on V2 PPO teacher trajectories.

Standalone trainer using HierarchicalStateEncoderV2 (8-dim inter, 7-dim intra)
and 11-dim action space.

Usage:
    cd /root/decision_transformer_slicing
    .pixi/envs/default/bin/python -m src.basic_apis.dt.train
"""
import os, sys, argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from src.basic_apis.dt.dataset import HierarchicalDTDataset
from src.basic_apis.dt.model import build_dt

NUM_SLICES = 5
ACTION_DIMS = [11] + [5] * NUM_SLICES + [3] * NUM_SLICES  # 11 heads


def train(
    dataset_dir="data/channel_generality/dt/dataset",
    save_dir="data/channel_generality/dt/dt_model",
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
    encoder_type="cross_attn",
    encoder_hidden_dim=None,
    encoder_num_heads=None,
    resume_from=None,
    total_epochs=None,
):
    device = torch.device(device_str)
    torch.set_num_threads(2)

    train_path = os.path.join(dataset_dir, "training")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading dataset from {train_path}")
    train_dataset = HierarchicalDTDataset(
        train_path, context_len=context_len, rtg_scale=rtg_scale, metadata_path=meta_path,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )

    act_dim = sum(ACTION_DIMS)
    from omegaconf import OmegaConf
    model_cfg = OmegaConf.create({
        "context_len": context_len, "embed_dim": embed_dim,
        "n_layer": n_layer, "n_head": n_head, "activation": "relu",
        "dropout": 0.1, "act_dim": act_dim,
    })

    model = build_dt(
        model_cfg=model_cfg, action_dims=ACTION_DIMS,
        inter_dim=8, intra_dim=7, embed_dim=embed_dim,
        encoder_type=encoder_type,
        encoder_hidden_dim=encoder_hidden_dim,
        encoder_num_heads=encoder_num_heads,
    ).to(device)

    # Only enable DataParallel when no specific GPU index is given (device_str == "cuda").
    # With "cuda:N" the user targets one GPU; wrapping in DataParallel would scatter to
    # ALL visible GPUs and OOM on already-occupied devices.
    _specific_gpu = ":" in device_str
    if not _specific_gpu and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    # Resume from a checkpoint if requested.
    # resume_from: path to a .pth file saved by a previous run.
    # total_epochs: if provided, the scheduler spans total_epochs (resumed + new).
    start_epoch = 0
    if resume_from is not None:
        resume_path = str(resume_from)
        print(f"Resuming from checkpoint: {resume_path}")
        raw = torch.load(resume_path, map_location=device)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(raw)
        # Infer which epoch we're resuming from by parsing the filename (e.g. epoch_30.pth).
        stem = Path(resume_path).stem
        if stem.startswith("epoch_"):
            start_epoch = int(stem.split("_")[1])
        print(f"  Loaded weights; resuming from epoch {start_epoch}")

    # total_epochs drives the cosine schedule so the resumed run sees the correct LR.
    _total = total_epochs if total_epochs is not None else (start_epoch + epochs)
    _remaining = _total - start_epoch  # how many epochs this call will train

    n_params = sum(p.numel() for p in model.parameters())
    n_enc_params = sum(p.numel() for p in model.state_encoder.parameters())
    print(
        f"Model params total={n_params:,}, encoder={n_enc_params:,}, "
        f"backbone={n_params - n_enc_params:,}, "
        f"act_dim={act_dim}, action_dims={ACTION_DIMS}"
    )

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=_total)
    # Fast-forward the scheduler to the correct position.
    for _ in range(start_epoch):
        scheduler.step()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

    scaler = torch.cuda.amp.GradScaler(enabled="cuda" in device_str)

    print(f"Training epochs {start_epoch+1}–{start_epoch+_remaining} "
          f"(total budget {_total}), batch_size={batch_size}, grad_accum={grad_accum}")

    for epoch in range(start_epoch, start_epoch + _remaining):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{start_epoch+_remaining}")
        for step, batch in enumerate(pbar):
            states = {
                "inter": batch["inter"].to(device),
                "intra": batch["intra"].to(device),
                "global": batch["global"].to(device),
            }
            actions_in = batch["actions"].to(device)
            actions_target = batch["action_targets"].to(device)
            rtg = batch["rtg"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)

            with torch.cuda.amp.autocast(enabled="cuda" in device_str):
                preds = model(
                    states=states, actions=actions_in, returns=rtg,
                    timesteps=timesteps, attention_mask=mask,
                )

                loss = 0.0
                preds_flat = preds.reshape(-1, preds.shape[-1])
                targets_flat = actions_target.reshape(-1, actions_target.shape[-1])

                start = 0
                for i, ds in enumerate(ACTION_DIMS):
                    end = start + ds
                    loss += loss_fn(preds_flat[:, start:end], targets_flat[:, i])
                    start = end

                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            losses.append(loss.item() * grad_accum)
            pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.3e}"})

        scheduler.step()
        avg = np.mean(losses)
        print(f"  Epoch {epoch+1}: avg_loss={avg:.4e}")

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            ckpt = os.path.join(save_dir, f"epoch_{epoch+1}.pth")
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state, ckpt)
            print(f"  Saved {ckpt}")

    final = os.path.join(save_dir, "final_dt.pth")
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, final)
    print(f"Training done. Final model: {final}")


def train_hydra(cfg):
    """Hydra entry point: called by channel_generality.py → train_dt mode."""
    tc = cfg.train_dt
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
        encoder_type=str(tc.model.encoder_type),
        encoder_hidden_dim=int(tc.model.encoder_hidden_dim) if tc.model.get("encoder_hidden_dim") is not None else None,
        encoder_num_heads=int(tc.model.encoder_num_heads) if tc.model.get("encoder_num_heads") is not None else None,
        resume_from=resume_from,
        total_epochs=total_epochs,
    )


def train_hydra_tiny(cfg):
    """Hydra entry point: called by channel_generality.py → train_dt_tiny mode."""
    tc = cfg.train_dt_tiny
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
        encoder_type=str(tc.model.encoder_type),
        encoder_hidden_dim=int(tc.model.encoder_hidden_dim) if tc.model.get("encoder_hidden_dim") is not None else None,
        encoder_num_heads=int(tc.model.encoder_num_heads) if tc.model.get("encoder_num_heads") is not None else None,
        resume_from=resume_from,
        total_epochs=total_epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="data/channel_generality/dt/dataset")
    parser.add_argument("--save_dir", type=str, default="data/channel_generality/dt/dt_model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--total_epochs", type=int, default=None,
                        help="Total epoch budget for scheduler (resumed + new); defaults to epochs")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--encoder", type=str, default="cross_attn",
        choices=["cross_attn", "mlp", "slice_attn"],
        help="State encoder type: cross_attn (original), mlp, slice_attn",
    )
    parser.add_argument(
        "--encoder_hidden_dim", type=int, default=None,
        help="Optional hidden dim for slice_attn encoder.",
    )
    parser.add_argument(
        "--encoder_num_heads", type=int, default=None,
        help="Optional attention heads for slice_attn encoder.",
    )
    args = parser.parse_args()

    train(
        dataset_dir=args.dataset, save_dir=args.save_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        embed_dim=args.embed_dim, n_layer=args.n_layer, n_head=args.n_head,
        grad_accum=args.grad_accum, device_str=args.device,
        encoder_type=args.encoder,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_num_heads=args.encoder_num_heads,
        resume_from=args.resume_from,
        total_epochs=args.total_epochs,
    )
