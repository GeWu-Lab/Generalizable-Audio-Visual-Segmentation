import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2  # type: ignore
import numpy as np
import torchmetrics
from collections import Counter
import sys
import random
sys.path.append('../modeling/')
sys.path.append('..')
sys.path.append('../../segment_anything/')
# print(sys.path)

import torch
import os
import logging
from datetime import datetime
import random
import numpy as np

# from segment_anything.loss.loss import BCE_Focal
from PIL import Image

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

set_seed(42)

import logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def log_agent(log_name='std'):

    # ch = logging.StreamHandler()  # 可输出至 Std-out
    # ch.setLevel(logging.INFO)  # or any other level
    # _logger.addHandler(ch)

    # log_file = open(save_print_path, 'w+')
    save_print_path = log_name
    fh = logging.FileHandler(save_print_path, encoding='utf-8')  # 输出至 文件
    # logger: save print into file
    _logger = logging.getLogger(log_name)
    _logger.setLevel(logging.DEBUG)  # process everything, even if everything isn't printed
    _logger.addHandler(fh)

    return _logger

# logger1 = log_agent('std1')
# logger1.info(f"1-Seeded tensor: {torch.rand(2, 2)}")


import matplotlib.pyplot as plt
import cv2



def save_mask(pred_masks, category_list=[], video_name_list=[], ver=None, tag=None, vid=None):
    save_base_path = f'/home/yaoting_wang/sam_proj/segment_anything/save_mask/{ver}_{tag}'
    
    mask_save_path = f'{save_base_path}/{vid}'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)
    # print(pred_masks.shape)
    pred_masks = pred_masks.unsqueeze(0)
    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
    
    # pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.view(-1, 1, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    
    bs = pred_masks.shape[0]
    # print('热点:', pred_masks.shape)  # [5, 1, 256, 256]
    if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
    for idx in range(bs):
        # category, video_name = category_list[idx], video_name_list[idx]
        # mask_save_path = os.path.join(save_base_path, category, video_name)
        
        one_mask = pred_masks[idx] # [5, 1, 224, 224]
        # video_name = 'output_mask.png'
        # output_name = "%s_%d.png"%(video_name, video_id)
        save_path = f'{mask_save_path}/{idx}.png'
        im = Image.fromarray(one_mask.squeeze()).convert('P')
        # print(save_path)
        im.save(save_path, format='PNG')

def remove_small_objects(image, min_size):
    '''Remove small objects from the binary image.
    
    Args:
        image: numpy array, binary image to be processed.
        min_size: int, the minimum area size (in pixels) to keep.
    
    Returns:
        The image without small objects.
    '''
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
    sizes = stats[:, -1]
    sizes[0] = 0  # remove the background component
    mask_sizes = sizes > min_size
    mask_sizes_vec = mask_sizes[labels]
    new_image = image.copy()
    new_image[~mask_sizes_vec] = 0
    return new_image

def remove_holes_and_smoothing(gray_image, kernel_size=3):
    '''Remove small objects from the binary image.
    
    Args:
        image: numpy array, binary image to be processed.
        kernel_size: int, size of the morphological kernel used by the operations.
    
    Returns:s
        The image with filled holes and smooth edges.
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # closed = opened
    return closed

def post_process(image):
    image = remove_small_objects(image, 800)
    image = remove_holes_and_smoothing(image)
    image = pp_jitter(image)
    return image

def pp_mask(mask):
    # plot-1
    h, w = mask.shape[-2:]
    mask = mask.detach().cpu().numpy()
    mask = mask.reshape(h, w) # * color.reshape(1, 1, -1)
    mask = mask.astype(np.uint8)
    
    gray_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    mask_pp = post_process(gray_mask)
    return mask_pp

def pp_mask_and_savefig(mask):
    mask_pp = pp_mask(mask)
    
    plt.axis('off')
    plt.tight_layout()
    # plt.imshow(mask_pp, cmap='Greys')
    plt.imshow(mask_pp, cmap='Greys')
    gray_fig = "../assets/test_01.png"
    plt.savefig(gray_fig, bbox_inches='tight', pad_inches=0)
    # input()
    return mask_pp, gray_fig


def get_loss_fn(fn, device):
    if fn == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif fn == 'focal':
        loss_fn = BCE_Focal()
    elif fn == 'dice':
        loss_fn = torchmetrics.Dice().to(device)
    elif fn == 'bce_dice':
        def bce_dice(_in, _tar):
            _bce = F.binary_cross_entropy_with_logits(_in, _tar)
            _dice = torchmetrics.Dice().to(device)(_in, _tar.int())
            return _bce + _dice
        loss_fn = bce_dice
    elif fn == 'avs':
        def _avs(_in, _tar):
            _focal_loss = BCE_Focal()(_in, _tar) * 20.0
            _dice_loss = torchmetrics.Dice().to(device)(_in, _tar.int())
            # print(f'_focal: {_focal_loss} | _dice: {_dice_loss}')
            return _focal_loss + _dice_loss
        loss_fn = _avs
    elif fn == 'sam':
        log_run.info('sam-loss')
        def sam_loss(_in, _tar):
            _focal = FocalLoss()(_in, _tar)
            _dice = torchmetrics.Dice().to(device)(_in, _tar.int())
            # _dice = DiceLoss()(_in, _tar.int())
            # print("sam:", _focal, _dice)
            _mse = ...  # TODO:
            return _focal * 20.0 + _dice  # ref: SAM.
        loss_fn = sam_loss  # focal + dice
    elif fn == 'bce':
        loss_fn = F.binary_cross_entropy_with_logits  # 'bce'
    elif fn == 'bbce':  # balanced bce
        def balanced_bce(_pred, _gt):
            # print(_gt.shape)
            num_neg = torch.sum((_gt.view(-1) == 0) + 0)
            num_pos = torch.sum((_gt.view(-1) == 1) + 0)
            # print(num_neg, num_pos)
            pos_weight = torch.tensor([num_neg / num_pos]).to(_pred.device)
            pos_weight = torch.tensor(100.0) if pos_weight == torch.inf else pos_weight
            # print(pos_weight)
            # _eps = 1e-10
            return F.binary_cross_entropy_with_logits(_pred, _gt, pos_weight=pos_weight)
        loss_fn = balanced_bce
    return loss_fn

def calc_miou(predict_mask, ground_truth_mask):
    intersection = torch.logical_and(predict_mask, ground_truth_mask).sum()
    union = torch.logical_or(predict_mask, ground_truth_mask).sum()
    if union == 0:
        iou = 1.0
    else:
        iou = float(intersection) / float(union)
    return iou
