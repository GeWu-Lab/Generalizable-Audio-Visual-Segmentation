"""Unified training/evaluation script for GAVS.

Supports v1s, v1m, v3 via --data flag.

Usage:
    # Single GPU
    python run.py --data v1m --tune_v 8 --train --val val --loss bce

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=8 run.py --data v1m --tune_v 8 --train --val val --loss bce
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

from config import args
from avs_model import AVSM
from build_sam import sam_model_registry

from utils.v1m import pyutils
from utils.v1m.utility import mask_iou, Eval_Fmeasure
from utils.utils import (
    set_seed, get_loss_fn, save_mask,
    setup_ddp, cleanup_ddp, is_ddp, is_main_process, get_local_rank, batch_to_device,
)

AUDIO_FROM = 'vggish_embs'

# Dataset split mapping per version
DATASET_SPLITS = {
    'v1s': {
        'module': 'dataset.avs_bench',
        'train': ('train', 'v1s'),
        'val': ('val', 'v1s'),
        'test': ('test', 'v1s'),
    },
    'v1m': {
        'module': 'dataset.avs_bench',
        'train': ('train', 'v1m'),
        'val': ('val', 'v1m'),
        'test': ('test', 'v1m'),
    },
    'v3': {
        'module': 'dataset.avs_bench_zsfs',
        'train': ('meta_v3_seen_train',),
        'val': ('meta_v3_seen_val',),
        'test': ('meta_v3_unseen',),
    },
}


def build_dataset(split, data_ver, feature_dir, model, audio_from):
    """Build dataset for the given split and version."""
    cfg = DATASET_SPLITS[data_ver]
    split_args = cfg[split]

    if data_ver in ('v1s', 'v1m'):
        from dataset.avs_bench import AVS
        return AVS(split_args[0], split_args[1], feature_dir, model=model, audio_from=audio_from)
    else:  # v3
        from dataset.avs_bench_zsfs import AVS
        return AVS(split_args[0], feature_dir, model=model, audio_from=audio_from)


def train(model, train_loader, optimizer, _ep, device):
    if is_main_process():
        print('train..')
    model.train()
    losses = []
    for batch_idx, batch_data in enumerate(train_loader):
        batch_data = batch_to_device(batch_data, device)
        img_recs, mask_recs, vid, _, _, _ = batch_data
        vid_preds, scores, loss_vid = model.forward(batch_data)

        loss_vid = torch.mean(torch.stack(loss_vid))
        optimizer.zero_grad()
        loss_vid.backward()
        optimizer.step()

        losses.append(loss_vid.item())
        if is_main_process():
            print(f'[loss-tr][{_ep}]: i: {batch_idx:04d} | loss={loss_vid.item():.08f} | score={scores} | {vid}', end='\r')

    return np.mean(losses)


def test(model, test_loader, device, mode='val'):
    model.eval()
    with torch.no_grad():
        losses = []

        for batch_idx, batch_data in enumerate(test_loader):
            batch_data = batch_to_device(batch_data, device)
            img_recs, mask_recs, vid, _, _, _ = batch_data
            vid_preds, scores, loss_vid = model.forward(batch_data)

            vid_preds_t = torch.stack(vid_preds, dim=0).squeeze()
            vid_masks_t = torch.stack(mask_recs, dim=0).squeeze()

            miou = mask_iou(vid_preds_t, vid_masks_t)
            avg_meter_miou.add({'miou': miou})

            F_score = Eval_Fmeasure(vid_preds_t, vid_masks_t, './logger', device=device)
            avg_meter_F.add({'F_score': F_score})

            loss_vid = torch.mean(torch.stack(loss_vid))
            losses.append(np.mean(loss_vid.item()))
            print(f'[loss-te]: i: {batch_idx:04d} | miou={miou:.03f} | F={F_score:.03f} | loss={loss_vid:.08f} | score={scores} | {vid}', end='\r')

    miou_epoch = (avg_meter_miou.pop('miou'))
    F_epoch = (avg_meter_F.pop('F_score'))
    return np.mean(losses), miou_epoch.item(), F_epoch


def run(model, device='cuda:0', data_ver='v1m'):
    max_miou, F_epoch = 0, 0
    miou_list, F_list = [], []

    feature_dir = './feature_extract'
    audio_from = AUDIO_FROM

    # Get the base model for dataset init (unwrap DDP if needed)
    base_model = model.module if is_ddp() else model
    sam_model = base_model.model_v

    train_dataset = build_dataset('train', data_ver, feature_dir, model=sam_model, audio_from=audio_from)
    val_dataset = build_dataset('val', data_ver, feature_dir, model=sam_model, audio_from=audio_from)
    test_dataset = build_dataset('test', data_ver, feature_dir, model=sam_model, audio_from=audio_from)

    # Use DistributedSampler for DDP
    train_sampler = DistributedSampler(train_dataset) if is_ddp() else None
    train_loader = DataLoader(
        train_dataset, batch_size=1,
        shuffle=(train_sampler is None),
        num_workers=4, pin_memory=True,
        sampler=train_sampler,
    )
    if args.val == 'val':
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    elif args.val == 'test' or args.val == 'test_in':
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Freeze all, then selectively enable LoRA params + audio projection
    for param in model.parameters():
        param.requires_grad = False

    trainable_keywords = ['lora_A', 'lora_B', 'audio_proj']
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True

    if is_main_process():
        tuned_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        print(f'Trainable parameters: {len(tuned_params)}')
        for name, _ in tuned_params:
            print(f"  requires_grad: {name}")

    # Optimizer
    params2 = [{'params': [p for name, p in model.named_parameters() if p.requires_grad], 'lr': 1e-3}]
    optimizer = torch.optim.AdamW(params2, lr=args.lr)

    # Training loop
    train_losses, val_losses, _lr_list = [], [], []
    for idx_ep in range(args.epochs):
        if is_main_process():
            print(f'[Epoch] {idx_ep}')
        currentDateAndTime = datetime.now().strftime("%y%m%d_%H_%M_%S_%f")

        # Sync epoch for DistributedSampler
        if train_sampler is not None:
            train_sampler.set_epoch(idx_ep)

        if args.train:
            model.train()
            loss_train = train(model, train_loader, optimizer, idx_ep, device)
            train_losses.append(loss_train)

            # Only rank 0 saves checkpoints
            if is_main_process():
                save_model = model.module if is_ddp() else model
                tag = f'{data_ver}/epoch_{idx_ep}.pth'
                save_path = f'../ckpt/{tag}'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(save_model.state_dict(), save_path)

        # Evaluation on rank 0 only (bypass DDP forward sync)
        if is_main_process() and (args.val == 'test_in' or args.val == 'val'):
            eval_model = model.module if is_ddp() else model
            eval_model.eval()
            loss_val, miou_epoch, F_epoch = test(eval_model, val_loader, device)
            miou_list.append(miou_epoch)
            F_list.append(F_epoch)
            print(f'val_loss: {loss_val} | val_miou: {miou_epoch} | val_f: {F_epoch}')

        if is_main_process():
            loss_train = loss_train if args.train else 0.0
            loss_val = loss_val if args.val else 0.0
            print(f'Epoch {idx_ep:02d} | train: {loss_train:.08f} | val: {loss_val:.08f} | mmiou: {max_miou} | F: {F_epoch} | Run at: {currentDateAndTime}')
            print(f'train-losses: {train_losses} | test-losses: {val_losses} | test-miou: {miou_list} | test-F: {F_list}')


if __name__ == '__main__':
    # DDP setup (no-op if not launched via torchrun)
    local_rank = setup_ddp()
    device = f'cuda:{local_rank}' if local_rank >= 0 else args.device
    torch.cuda.set_device(device)

    data_ver = args.data

    set_seed(42)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    if is_main_process():
        print(f'----- Dataset: {data_ver} | Audio: {AUDIO_FROM} !')
        checkpoint_1 = args.checkpoint
        print(f'Run with checkpoint: {checkpoint_1}')

    checkpoint_1 = args.checkpoint
    loss_fn = get_loss_fn(args.loss, device=device)
    optim_config = {
        'ver': data_ver,
        'loss': loss_fn,
        'tune_v': args.tune_v,
    }

    sam_avs = sam_model_registry[args.model_type](checkpoint_1).to(device)

    # Enable LoRA on selected layers
    n_enc = sam_avs.image_encoder.enable_lora(tune_v=args.tune_v)
    n_dec = sam_avs.mask_decoder.transformer.enable_lora()
    if is_main_process():
        print(f'LoRA enabled: {n_enc} encoder layers, {n_dec} decoder layers')

    avs = AVSM(model_v=sam_avs, model_t=None, config=optim_config).to(device)

    # Wrap with DDP if distributed
    if is_ddp():
        avs = DDP(avs, device_ids=[local_rank], find_unused_parameters=True)

    torch.multiprocessing.set_start_method('spawn', force=True)
    if is_main_process():
        print('use device:', device, f'(world_size={torch.distributed.get_world_size()})' if is_ddp() else '(single GPU)')
    run(avs, device=device, data_ver=data_ver)

    cleanup_ddp()
