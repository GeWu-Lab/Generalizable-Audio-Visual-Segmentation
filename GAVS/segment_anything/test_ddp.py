"""Minimal DDP smoke test for GAVS with LoRA.

Verifies: model init → LoRA enable → DDP wrap → forward → backward → gradient sync.
Uses synthetic data (no real AVS dataset needed).

Launch:
  torchrun --nproc_per_node=8 test_ddp.py --tune_v 8 --train --loss bce
  torchrun --nproc_per_node=2 test_ddp.py --tune_v 8 --train --loss bce   # quick 2-GPU test
"""

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import os
import numpy as np

from config import args
from avs_model import AVSM
from build_sam import sam_model_registry
from utils.utils import (
    set_seed, get_loss_fn,
    setup_ddp, cleanup_ddp, is_ddp, is_main_process, get_local_rank, batch_to_device,
)


class SyntheticAVS(Dataset):
    """Synthetic dataset that mimics AVS batch structure with random data."""

    def __init__(self, num_samples=32, num_frames=5, img_size=1024, model=None):
        self.num_samples = num_samples
        self.num_frames = num_frames
        self.img_size = img_size
        # Extract preprocess params from model (same as real dataset)
        self.pixel_mean = model.pixel_mean.cpu().clone()
        self.pixel_std = model.pixel_std.cpu().clone()

    def __len__(self):
        return self.num_samples

    def _preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        img_recs = []
        mask_recs = []

        # Synthetic audio: [num_frames, 128]
        feat_aud = torch.randn(self.num_frames, 128)

        for f_idx in range(self.num_frames):
            # Synthetic image: [1, 3, 1024, 1024] preprocessed
            raw_img = torch.randint(0, 256, (1, 3, 768, 1024), dtype=torch.float32)
            input_image = self._preprocess(raw_img).squeeze()

            # Synthetic pre-extracted image embedding: [256, 64, 64]
            image_embed = torch.randn(256, 64, 64)

            transformed_data = {
                'image': input_image,
                'input_size': (768, 1024),
                'original_size': (720, 1280),
                'image_embed': image_embed,
                'audio': feat_aud[f_idx],
            }
            img_recs.append(transformed_data)

            # Synthetic binary mask: [1, 1, 720, 1280]
            gt_mask = torch.zeros(1, 1, 720, 1280, dtype=torch.float32)
            gt_mask[:, :, 200:500, 300:900] = 1.0  # rectangle
            mask_recs.append(gt_mask)

        vid = f'synthetic_vid_{idx}'
        category = 'test'
        return img_recs, mask_recs, vid, category, feat_aud, feat_aud


def main():
    local_rank = setup_ddp()
    device = f'cuda:{local_rank}' if local_rank >= 0 else args.device
    torch.cuda.set_device(device)

    set_seed(42)
    num_epochs = 2

    if is_main_process():
        print(f'=== DDP Smoke Test ===')
        print(f'World size: {torch.distributed.get_world_size() if is_ddp() else 1}')
        print(f'Device: {device}')

    # Build model
    checkpoint_path = args.checkpoint
    loss_fn = get_loss_fn(args.loss, device=device)
    optim_config = {
        'ver': 'v1m',
        'loss': loss_fn,
        'tune_v': args.tune_v,
    }

    sam_avs = sam_model_registry[args.model_type](checkpoint_path).to(device)
    n_enc = sam_avs.image_encoder.enable_lora(tune_v=args.tune_v)
    n_dec = sam_avs.mask_decoder.transformer.enable_lora()
    if is_main_process():
        print(f'LoRA enabled: {n_enc} encoder layers, {n_dec} decoder layers')

    model = AVSM(model_v=sam_avs, model_t=None, config=optim_config).to(device)

    # Wrap with DDP
    if is_ddp():
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Synthetic dataset
    base_model = model.module if is_ddp() else model
    dataset = SyntheticAVS(num_samples=32, num_frames=5, model=base_model.model_v)

    train_sampler = DistributedSampler(dataset) if is_ddp() else None
    train_loader = DataLoader(
        dataset, batch_size=1,
        shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True,
        sampler=train_sampler,
    )

    # Freeze all, then enable LoRA + audio_proj
    for param in model.parameters():
        param.requires_grad = False

    trainable_keywords = ['lora_A', 'lora_B', 'audio_proj']
    for name, param in model.named_parameters():
        if any(kw in name for kw in trainable_keywords):
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process():
        print(f'Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)')

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    # Training loop
    if is_main_process():
        print(f'\n--- Training {num_epochs} epochs ---')

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_losses = []

        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_to_device(batch_data, device)
            vid_preds, scores, loss_vid = model.forward(batch_data)

            loss = torch.mean(torch.stack(loss_vid))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if is_main_process() and batch_idx % 4 == 0:
                print(f'  [Epoch {epoch}] batch {batch_idx:03d} | loss={loss.item():.6f}')

        mean_loss = np.mean(epoch_losses)
        if is_main_process():
            print(f'Epoch {epoch} | mean_loss={mean_loss:.6f} | batches={len(epoch_losses)}')

    # Verify gradient sync: check that a LoRA param has the same value across ranks
    if is_ddp():
        for name, param in model.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                # Gather param from all ranks to rank 0
                gathered = [torch.zeros_like(param) for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather(gathered, param.data)
                if is_main_process():
                    all_same = all(torch.allclose(gathered[0], g, atol=1e-6) for g in gathered[1:])
                    print(f'\nGradient sync check ({name}): {"PASS" if all_same else "FAIL"}')
                break

    if is_main_process():
        print('\n=== DDP Smoke Test PASSED ===')

    cleanup_ddp()


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
