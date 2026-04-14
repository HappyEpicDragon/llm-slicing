#!/bin/bash
set -euo pipefail
cd /root/decision_transformer_slicing


echo "=== k1_2 gpu=1 train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/sensitivity/s3_exhaustive/k1_2 \
    train_dt_v2_tiny.save_dir=data/channel_generality/sensitivity/s3_exhaustive/k1_2/model \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda > data/channel_generality/sensitivity/logs/exhaustive/k1_2.log 2>&1

echo "=== k1_2 gpu=1 eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/sensitivity/s3_exhaustive/k1_2/model/final_dt_v2.pth \
    test_dt_v2_tiny.meta_path=data/channel_generality/sensitivity/s3_exhaustive/k1_2/metadata.json \
    test_dt_v2_tiny.model.context_len=20 \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root=data/channel_generality/sensitivity/s3_exhaustive/k1_2/eval \
    test_dt_v2_tiny.device=cuda >> data/channel_generality/sensitivity/logs/exhaustive/k1_2.log 2>&1

echo "=== k1_2 DONE $(date) ==="


echo "=== k2_0_3 gpu=1 train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/sensitivity/s3_exhaustive/k2_0_3 \
    train_dt_v2_tiny.save_dir=data/channel_generality/sensitivity/s3_exhaustive/k2_0_3/model \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda > data/channel_generality/sensitivity/logs/exhaustive/k2_0_3.log 2>&1

echo "=== k2_0_3 gpu=1 eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/sensitivity/s3_exhaustive/k2_0_3/model/final_dt_v2.pth \
    test_dt_v2_tiny.meta_path=data/channel_generality/sensitivity/s3_exhaustive/k2_0_3/metadata.json \
    test_dt_v2_tiny.model.context_len=20 \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root=data/channel_generality/sensitivity/s3_exhaustive/k2_0_3/eval \
    test_dt_v2_tiny.device=cuda >> data/channel_generality/sensitivity/logs/exhaustive/k2_0_3.log 2>&1

echo "=== k2_0_3 DONE $(date) ==="


echo "=== k2_1_4 gpu=1 train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/sensitivity/s3_exhaustive/k2_1_4 \
    train_dt_v2_tiny.save_dir=data/channel_generality/sensitivity/s3_exhaustive/k2_1_4/model \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda > data/channel_generality/sensitivity/logs/exhaustive/k2_1_4.log 2>&1

echo "=== k2_1_4 gpu=1 eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/sensitivity/s3_exhaustive/k2_1_4/model/final_dt_v2.pth \
    test_dt_v2_tiny.meta_path=data/channel_generality/sensitivity/s3_exhaustive/k2_1_4/metadata.json \
    test_dt_v2_tiny.model.context_len=20 \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root=data/channel_generality/sensitivity/s3_exhaustive/k2_1_4/eval \
    test_dt_v2_tiny.device=cuda >> data/channel_generality/sensitivity/logs/exhaustive/k2_1_4.log 2>&1

echo "=== k2_1_4 DONE $(date) ==="


echo "=== k3_0_1_3 gpu=1 train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/sensitivity/s3_exhaustive/k3_0_1_3 \
    train_dt_v2_tiny.save_dir=data/channel_generality/sensitivity/s3_exhaustive/k3_0_1_3/model \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda > data/channel_generality/sensitivity/logs/exhaustive/k3_0_1_3.log 2>&1

echo "=== k3_0_1_3 gpu=1 eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/sensitivity/s3_exhaustive/k3_0_1_3/model/final_dt_v2.pth \
    test_dt_v2_tiny.meta_path=data/channel_generality/sensitivity/s3_exhaustive/k3_0_1_3/metadata.json \
    test_dt_v2_tiny.model.context_len=20 \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root=data/channel_generality/sensitivity/s3_exhaustive/k3_0_1_3/eval \
    test_dt_v2_tiny.device=cuda >> data/channel_generality/sensitivity/logs/exhaustive/k3_0_1_3.log 2>&1

echo "=== k3_0_1_3 DONE $(date) ==="


echo "=== k3_0_3_4 gpu=1 train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/sensitivity/s3_exhaustive/k3_0_3_4 \
    train_dt_v2_tiny.save_dir=data/channel_generality/sensitivity/s3_exhaustive/k3_0_3_4/model \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda > data/channel_generality/sensitivity/logs/exhaustive/k3_0_3_4.log 2>&1

echo "=== k3_0_3_4 gpu=1 eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/sensitivity/s3_exhaustive/k3_0_3_4/model/final_dt_v2.pth \
    test_dt_v2_tiny.meta_path=data/channel_generality/sensitivity/s3_exhaustive/k3_0_3_4/metadata.json \
    test_dt_v2_tiny.model.context_len=20 \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root=data/channel_generality/sensitivity/s3_exhaustive/k3_0_3_4/eval \
    test_dt_v2_tiny.device=cuda >> data/channel_generality/sensitivity/logs/exhaustive/k3_0_3_4.log 2>&1

echo "=== k3_0_3_4 DONE $(date) ==="


echo "=== k3_2_3_4 gpu=1 train start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim train_dt_v2_tiny \
    train_dt_v2_tiny.grid_id=b2_e2 \
    train_dt_v2_tiny.dataset_dir=data/channel_generality/sensitivity/s3_exhaustive/k3_2_3_4 \
    train_dt_v2_tiny.save_dir=data/channel_generality/sensitivity/s3_exhaustive/k3_2_3_4/model \
    train_dt_v2_tiny.model.context_len=20 \
    train_dt_v2_tiny.model.embed_dim=32 \
    train_dt_v2_tiny.model.n_layer=2 \
    train_dt_v2_tiny.model.n_head=2 \
    train_dt_v2_tiny.model.encoder_type=slice_attn \
    train_dt_v2_tiny.model.encoder_hidden_dim=32 \
    train_dt_v2_tiny.model.encoder_num_heads=2 \
    train_dt_v2_tiny.optimizer.epochs=30 \
    train_dt_v2_tiny.optimizer.batch_size=1024 \
    train_dt_v2_tiny.optimizer.lr=1e-4 \
    train_dt_v2_tiny.device=cuda > data/channel_generality/sensitivity/logs/exhaustive/k3_2_3_4.log 2>&1

echo "=== k3_2_3_4 gpu=1 eval start $(date) ==="
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 /root/.pixi/bin/pixi run sim test_dt_v2_tiny \
    test_dt_v2_tiny.grid_id=b2_e2 \
    test_dt_v2_tiny.model_path=data/channel_generality/sensitivity/s3_exhaustive/k3_2_3_4/model/final_dt_v2.pth \
    test_dt_v2_tiny.meta_path=data/channel_generality/sensitivity/s3_exhaustive/k3_2_3_4/metadata.json \
    test_dt_v2_tiny.model.context_len=20 \
    test_dt_v2_tiny.model.embed_dim=32 \
    test_dt_v2_tiny.model.n_layer=2 \
    test_dt_v2_tiny.model.n_head=2 \
    test_dt_v2_tiny.encoder_type=slice_attn \
    test_dt_v2_tiny.model.encoder_hidden_dim=32 \
    test_dt_v2_tiny.model.encoder_num_heads=2 \
    'test_dt_v2_tiny.test_scenarios=[5,6,7,8,9]' \
    'test_dt_v2_tiny.test_seeds=[0]' \
    test_dt_v2_tiny.n_episodes=20 \
    test_dt_v2_tiny.init_episode=0 \
    test_dt_v2_tiny.max_episode=100 \
    test_dt_v2_tiny.target_rtg=0 \
    test_dt_v2_tiny.save_root=data/channel_generality/sensitivity/s3_exhaustive/k3_2_3_4/eval \
    test_dt_v2_tiny.device=cuda >> data/channel_generality/sensitivity/logs/exhaustive/k3_2_3_4.log 2>&1

echo "=== k3_2_3_4 DONE $(date) ==="


echo "=== GPU 1 ALL JOBS DONE $(date) ==="
