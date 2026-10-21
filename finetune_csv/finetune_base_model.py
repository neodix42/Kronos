import os
import sys
import json
import time
import pickle
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from time import gmtime, strftime
import logging
from logging.handlers import RotatingFileHandler
import datetime
from contextlib import contextmanager
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import Kronos, KronosTokenizer
from finetune_csv.config_loader import CustomFinetuneConfig
from finetune_csv.data_utils import FEATURES, load_candles, split_bounds, contiguous_window_starts


class CustomKlineDataset(Dataset):
    """Leakage-free windows, split before sampling and normalized from context only."""

    def __init__(self, data_path, data_type='train', lookback_window=488, predict_window=24,
                 clip=5.0, seed=42, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
                 interval=None, data_sha256=None):
        self.data_path = data_path
        self.data_type = data_type
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.window = lookback_window + predict_window
        self.clip = clip
        self.seed = seed
        self.feature_list = FEATURES
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']
        config = {'data': {'data_path': data_path, 'interval': interval,
                          'train_ratio': train_ratio, 'val_ratio': val_ratio,
                          'test_ratio': test_ratio, 'sha256': data_sha256}}
        frame = load_candles(config)
        train_end, val_end = split_bounds(frame, config)
        bounds = {'train': (0, train_end), 'val': (train_end, val_end),
                  'test': (val_end, len(frame))}
        if data_type not in bounds:
            raise ValueError(f'Unknown dataset split: {data_type}')
        left, right = bounds[data_type]
        self.split_start, self.split_end = left, right
        self.data = frame.iloc[left:right].copy().reset_index(drop=True)
        self.timestamps = self.data['timestamps'].copy()
        for name in self.time_feature_list:
            self.data[name] = getattr(self.data['timestamps'].dt, name)
        self.starts = contiguous_window_starts(self.data, self.window, frame.attrs['interval'])
        self.n_samples = len(self.starts)
        possible = max(0, len(self.data) - self.window + 1)
        self.skipped_gap_windows = possible - self.n_samples
        self.values = self.data[self.feature_list].to_numpy(dtype=np.float32)
        self.time_values = self.data[self.time_feature_list].to_numpy(dtype=np.float32)
        print(f'[{data_type.upper()}] {len(self.data)} raw rows, {self.n_samples} windows; '
              f'{self.skipped_gap_windows} windows excluded across missing candle slots')
        if not self.n_samples:
            minimum = int(np.ceil(self.window / min(train_ratio, val_ratio)))
            raise ValueError(f'{data_type} split has no contiguous {self.window}-candle window. '
                             f'Use at least {minimum} rows with these split ratios, or download '
                             'more history / reduce window sizes; missing candle slots can require more.')

    def set_epoch_seed(self, epoch):
        self.current_epoch = epoch

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(idx)
        start = self.starts[idx]
        x = self.values[start:start + self.window].copy()
        context = x[:self.lookback_window]
        mean, std = context.mean(axis=0), context.std(axis=0)
        x = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)
        stamps = self.time_values[start:start + self.window].copy()
        return torch.from_numpy(x), torch.from_numpy(stamps)


def setup_logging(exp_name: str, log_dir: str, rank: int = 0) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(f"basemodel_training_rank_{rank}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    log_file = os.path.join(log_dir, f"basemodel_training_rank_{rank}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    console_handler = None
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    if console_handler is not None:
        console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    if console_handler is not None:
        logger.addHandler(console_handler)
    
    logger.info(f"=== Basemodel Training Started ===")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Rank: {rank}")
    logger.info(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return logger


def create_dataloaders(config):
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print("Creating data loaders...")
    
    train_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type='train',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        interval=getattr(config, "interval", None),
        data_sha256=config.loader.get("data.sha256") if hasattr(config, "loader") else None
    )
    
    val_dataset = CustomKlineDataset(
        data_path=config.data_path,
        data_type='val',
        lookback_window=config.lookback_window,
        predict_window=config.predict_window,
        clip=config.clip,
        seed=config.seed + 1,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        interval=getattr(config, "interval", None),
        data_sha256=config.loader.get("data.sha256") if hasattr(config, "loader") else None
    )
    
    use_ddp = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, drop_last=False) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=val_sampler
    )
    
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    return train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler


@contextmanager
def causal_validation(model):
    """Keep teacher-forced validation causal without changing inference behavior."""
    raw_model = model.module if isinstance(model, DDP) else model
    attention = raw_model.dep_layer.cross_attn
    training, dropout = attention.training, attention.attn_dropout_p
    # This attention module selects its causal mask through its training flag.
    # Its child dropout modules remain in evaluation mode.
    attention.training, attention.attn_dropout_p = True, 0.0
    try:
        yield
    finally:
        attention.training, attention.attn_dropout_p = training, dropout


def train_model(model, tokenizer, device, config, save_dir, logger):
    logger.info("Starting training...")
    use_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    
    train_loader, val_loader, train_dataset, val_dataset, train_sampler, val_sampler = create_dataloaders(config)
    tokenizer.eval()
    tokenizer.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=config.basemodel_epochs,
        pct_start=0.03,
        div_factor=10
    )
    
    if use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    best_val_loss = float('inf')
    batch_idx_global = 0
    
    for epoch in range(config.basemodel_epochs):
        epoch_start_time = time.time()
        model.train()
        
        train_dataset.set_epoch_seed(epoch * 10000)
        val_dataset.set_epoch_seed(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                           use_teacher_forcing=True, s1_targets=token_out[0])
            # Supervise only the actual forecast horizon. Earlier context-token
            # losses would use normalization statistics unavailable at those times.
            target_start = config.lookback_window - 1
            loss, s1_loss, s2_loss = (model.module if use_ddp else model).head.compute_loss(
                logits[0][:, target_start:], logits[1][:, target_start:],
                token_out[0][:, target_start:], token_out[1][:, target_start:])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_((model.module if use_ddp else model).parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx_global + 1) % config.log_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_msg = (f"[Epoch {epoch+1}/{config.basemodel_epochs}, Step {batch_idx+1}/{len(train_loader)}] "
                          f"LR: {lr:.6f}, Loss: {loss.item():.4f}")
                logger.info(log_msg)
                if rank == 0:
                    print(log_msg)
            
            batch_idx_global += 1
        
        model.eval()
        raw_model = model.module if use_ddp else model
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad(), causal_validation(model):
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
                
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                               use_teacher_forcing=True, s1_targets=token_out[0])
                target_start = config.lookback_window - 1
                loss, _, _ = raw_model.head.compute_loss(
                    logits[0][:, target_start:], logits[1][:, target_start:],
                    token_out[0][:, target_start:], token_out[1][:, target_start:])
                
                val_loss += loss.item()
                val_batches += 1
        
        if use_ddp:
            tensor_sum = torch.tensor([epoch_train_loss, train_batches, val_loss, val_batches], dtype=torch.float64, device=device)
            dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)
            epoch_train_loss_all = tensor_sum[0].item()
            train_batches_all = int(tensor_sum[1].item())
            val_loss_all = tensor_sum[2].item()
            val_batches_all = int(tensor_sum[3].item())
            avg_train_loss = (epoch_train_loss_all / train_batches_all) if train_batches_all > 0 else 0.0
            avg_val_loss = (val_loss_all / val_batches_all) if val_batches_all > 0 else 0.0
        else:
            avg_train_loss = epoch_train_loss / train_batches if train_batches > 0 else 0
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        epoch_time = time.time() - epoch_start_time
        epoch_summary = (f"\n--- Epoch {epoch+1}/{config.basemodel_epochs} Summary ---\n"
                       f"Training Loss: {avg_train_loss:.4f}\n"
                       f"Validation Loss: {avg_val_loss:.4f}\n"
                       f"Epoch Time: {epoch_time:.2f} seconds\n")
        logger.info(epoch_summary)
        if rank == 0:
            print(epoch_summary)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if rank == 0:
                model_save_path = os.path.join(save_dir, "best_model")
                os.makedirs(model_save_path, exist_ok=True)
                (model.module if use_ddp else model).save_pretrained(model_save_path)
                save_msg = f"Best model saved to: {model_save_path} (validation loss: {best_val_loss:.4f})"
                logger.info(save_msg)
                print(save_msg)
    
    return best_val_loss


def main():
    import argparse
    from finetune_csv.train_sequential import SequentialTrainer
    parser = argparse.ArgumentParser(description='Train the Kronos predictor from CSV')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    trainer = SequentialTrainer(args.config)
    trainer.config.train_tokenizer = False
    trainer.config.train_basemodel = True
    if not trainer.run_training():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
