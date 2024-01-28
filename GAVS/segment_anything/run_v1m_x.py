import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import threshold, normalize
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

from config import args
from avs_model import AVSM
from dataset.avs_bench import AVS
# from modeling.sam import Sam
# from segment_anything import SamAutomaticMaskGenerator
from build_sam import sam_model_registry
# from segment_anything.loss.loss import FocalLoss, DiceLoss

# from segment_anything.utils.utils import log_agent
from utils.v1m import pyutils
from utils.v1m.utility import mask_iou, Eval_Fmeasure
# from utils.v1m.loss import IouSemanticAwareLoss, F1_IoU_BCELoss
from utils.utils import set_seed, get_loss_fn, save_mask
# from utils.transforms import ResizeLongestSide
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
# from collections import defaultdict

DATA = 'v1m'
AUDIO_FROM = 'vggish_embs'
# AUDIO_FROM = 'aud_ib'
# AUDIO_FROM = 'aud_emb'

def train(model, train_loader, optimizer, _ep):
    # for m, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(m)
    # input()
    print('train..')
    model.train()
    losses = []
    for batch_idx, batch_data in enumerate(train_loader):
        img_recs, mask_recs, vid, _, _, _ = batch_data
        vid_preds, scores, loss_vid = model.forward(batch_data)

        loss_vid = torch.mean(torch.stack(loss_vid))
        # print(f'[train] Video loss: {loss_vid}')
        optimizer.zero_grad()
        loss_vid.backward()
        optimizer.step()

        losses.append(loss_vid.item())  # collect videos.
        print(f'[loss-tr][{_ep}]: i: {batch_idx:04d} | loss={loss_vid.item():.08f} | score={scores} | {vid}', end='\r')
        # print(f'[loss-tr][{_ep}]: i: {batch_idx:04d} | loss={loss_vid.item():.08f} | score={scores} | {vid}')

    return np.mean(losses)


def test(model, test_loader, mode='val'):
    model.eval()
    with torch.no_grad():
        losses = []

        for batch_idx, batch_data in enumerate(test_loader):
            img_recs, mask_recs, vid, _, _, _ = batch_data
            vid_preds, scores, loss_vid = model.forward(batch_data)

            vid_preds_t = torch.stack(vid_preds, dim=0).squeeze()  # [5, 720, 1280] = [1*frames, H, W]
            vid_masks_t = torch.stack(mask_recs, dim=0).squeeze()
                    
            miou = mask_iou(vid_preds_t, vid_masks_t)  # mIoU
            avg_meter_miou.add({'miou': miou})

            F_score = Eval_Fmeasure(vid_preds_t, vid_masks_t, './logger', device=args.device)  # F_score
            avg_meter_F.add({'F_score': F_score})

            loss_vid = torch.mean(torch.stack(loss_vid))
            losses.append(np.mean(loss_vid.item()))
            # print(score)
            print(f'[loss-te]: i: {batch_idx:04d} | miou={miou:.03f} | F={F_score:.03f} | loss={loss_vid:.08f} | score={scores} | {vid}', end='\r')

    miou_epoch = (avg_meter_miou.pop('miou'))
    F_epoch = (avg_meter_F.pop('F_score'))
    return np.mean(losses), miou_epoch.item(), F_epoch


def run(model, device='cuda:1', ckpt_dir='./none', data_ver=DATA):
    max_miou, F_epoch = 0, 0
    miou_list, F_list = [], []

    ckpt_dir = f'./checkpoint/{data_ver}/run_v_sa/'
    feature_dir = './feature_extract'
    audio_from = AUDIO_FROM
    train_dataset = AVS('train', data_ver, feature_dir, device=device, model=model.model_v, audio_from=audio_from)   
    val_dataset = AVS('val', data_ver, feature_dir, device=device, model=model.model_v, audio_from=audio_from)  # val
    test_dataset = AVS('test', data_ver, feature_dir, device=device, model=model.model_v, audio_from=audio_from)  # test

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=False)
    if args.val == 'val':
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    elif args.val == 'test' or args.val == 'test_in':
        val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    # tuned params:
    params_names = [
        # 'sparse_proj',
        'adapter_tf',
        'audio_proj',
        'llm_proj',
        'adapter_v',
        'seq_proj',
    ]

    # fine-tune model
    tuned_num = 0
    for name, param in model.named_parameters():
        param.requires_grad = False
        for _n in params_names:
            if _n in name:   
                param.requires_grad = True   
                tuned_num += 1

    for name, param in model.model_v.image_encoder.blocks.named_parameters():
        param.requires_grad = False
        tune_blk = args.tune_v
        blk_id = name.split('.')[0]
        if int(blk_id) >= tune_blk and 'adapter_v' in name:
            param.requires_grad = True
        if int(blk_id) >= tune_blk+1 and 'norm' in name:
            param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Requires_grad:", name)
    # input()

    # Optimizer & Loss
    params2 = [{'params': [p for name, p in model.named_parameters() if p.requires_grad], 'lr': 1e-3}]
    optimizer = torch.optim.AdamW(params2, lr=args.lr)

    # train and val/test the model
    model = model.to(device)
    train_losses, val_losses, _lr_list = [], [], []
    for idx_ep in range(args.epochs):
        print(f'[Epoch] {idx_ep}')
        currentDateAndTime = datetime.now().strftime("%y%m%d_%H_%M_%S_%f")

        if args.train:
            model.train()
            loss_train = train(model, train_loader, optimizer, idx_ep)
            train_losses.append(loss_train)

            tag = f'no_tune'
            tag = f'{tag}/epoch_{idx_ep}.pth'
            save_path = f'../ckpt/{tag}'
            torch.save(model.state_dict(), save_path)

        if args.val == 'test_in' or args.val == 'val':
            model.eval()
            loss_val, miou_epoch, F_epoch = test(model, val_loader)
            miou_list.append(miou_epoch)
            F_list.append(F_epoch)
            print(f'val_loss: {loss_val} | val_miou: {miou_epoch} | val_f: {F_epoch}')

        loss_train = loss_train if args.train else 0.0
        loss_val = loss_val if args.val else 0.0
        print(f'Epoch {idx_ep:02d} | train: {loss_train:.08f} | val: {loss_val:.08f} | mmiou: {max_miou} | F: {F_epoch} | Run at: {currentDateAndTime}')
        print(f'train-losses: {train_losses} | test-losses: {val_losses} | test-miou: {miou_list} | test-F: {F_list}')


if __name__ == '__main__':
    set_seed(42)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    print(f'----- NOTE: using audio from {AUDIO_FROM} !')
    checkpoint_1 = args.checkpoint
    input(f' --- Run with checkpoint: {checkpoint_1}, continue? ---')

    loss_fn = get_loss_fn(args.loss, device='cuda')
    optim_config = {
        'ver': 'v1m',
        'loss': loss_fn,
        'device': 'cuda',
        'tune_v': args.tune_v,
    }

    sam_avs = sam_model_registry[args.model_type](checkpoint_1).to(args.device)  # strict=False
    avs = AVSM(model_v=sam_avs, model_t=None, config=optim_config).to(args.device)
    # pretrained = torch.load(checkpoint_1)
    # avs.load_state_dict(pretrained)

    torch.multiprocessing.set_start_method('spawn', force=True)
    print('use device:', args.device)
    run(avs, device=args.device)

