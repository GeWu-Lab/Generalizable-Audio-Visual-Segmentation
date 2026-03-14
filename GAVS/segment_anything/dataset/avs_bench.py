import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb

import sys
import os
import random

sys.path.append('../modeling/')
sys.path.append('..')
sys.path.append('../../segment_anything/')
# print(sys.path)
from utils.utils import log_agent
from utils.transforms import ResizeLongestSide
from torchvision import transforms
from collections import defaultdict
import cv2
from PIL import Image

import pickle as pkl

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

class AVS(Dataset):
    def __init__(self, split='train', ver='v2', feature_dir='', model=None, device=None, audio_from=None):
        # Extract preprocessing params from model (not stored to avoid pickling overhead in workers)
        self.img_size = model.image_encoder.img_size
        self.pixel_mean = model.pixel_mean.cpu().clone()
        self.pixel_std = model.pixel_std.cpu().clone()
        self.transform = ResizeLongestSide(self.img_size)
        self.ver = ver
        self.feature_base_path = feature_dir
        _dir = os.path.dirname(os.path.abspath(__file__))
        _repo = os.path.dirname(os.path.dirname(os.path.dirname(_dir)))  # GAVS/GAVS/..
        self.data_base_path = os.path.join(_repo, 'data', 'AVS', ver)
        self.audio_from = audio_from
        meta_path = os.path.join(_repo, 'data', 'AVS', 'metadata.csv')
        metadata = pd.read_csv(meta_path, header=0)
        sub_data = metadata[metadata['label'] == ver]  # v1s set
        # sub_data_train = sub_data[sub_data['split'] == 'train']
        # sub_data_test = sub_data[sub_data['split'] == 'test']
        # sub_data_val = sub_data[sub_data['split'] == 'val']
        self.split = split
        self.metadata = sub_data[sub_data['split'] == split]  # split= train,test,val.
        # print(self.metadata)
        # input()

        self.audio = None
        self.images = None

        self.frame_num = 10 if ver == 'v2' else 5
        # self.frame_num = 2
        # print(f'[dataset] self.frame_num: {self.frame_num}')
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

        # self.logger = log_agent('dataset.log')

        self.data_path = self.data_base_path
        self.feat_path = os.path.join(_dir, '..', 'feature_extract')

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input (CPU-safe)."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid, category = df_one_video['uid'], df_one_video['a_obj']  # uid for vid.

        img_recs = []
        mask_recs = []

        feat_aud_p = f'{self.feat_path}/{self.ver}_vggish_embs/{vid}.npy'
        feat_aud = torch.from_numpy(np.load(feat_aud_p)).squeeze().detach()

        FN = self.frame_num
        for _idx in range(FN):  # set frame_num as the batch_size
            path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'  # image

            # Load pre-extracted image embedding if available (used when tune_v >= 12)
            if self.ver == 'v1s':
                feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f{_idx}.npy'
            else:
                feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f{_idx}.pth'

            if os.path.exists(feat_img_p):
                if feat_img_p.endswith('.npy'):
                    image_embed = torch.from_numpy(np.load(feat_img_p)).squeeze()
                else:
                    image_embed = torch.load(feat_img_p, map_location='cpu').squeeze()
            else:
                image_embed = torch.empty(0)  # placeholder; encoder runs on-the-fly

            # data
            transformed_data = defaultdict(dict)
            image = cv2.imread(path_frame)
            # image = cv2.resize(image, (720, 1280))

            input_image = self.transform.apply_image(image)

            input_image_torch = torch.as_tensor(input_image)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            # prepare for input
            input_image = self._preprocess(transformed_image)
            # print(image.shape)
            original_image_size = (image.shape[0], image.shape[1])  # H x W
            input_size = tuple(transformed_image.shape[-2:])

            # embedding
            audio_embed = feat_aud[_idx].squeeze()

            # dict input
            transformed_data['image'] = input_image.squeeze()
            transformed_data['input_size'] = input_size
            transformed_data['original_size'] = original_image_size
            transformed_data['image_embed'] = image_embed
            transformed_data['audio'] = audio_embed
            # transformed_data['engine_input'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # mask label
            _idx_mask = 0 if self.split == 'train' and self.ver == 'v1s' else _idx
            path_mask = f'{self.data_path}/{vid}/labels_rgb/{_idx_mask}.png'
            mask_cv2 = cv2.imread(path_mask)
            # mask_cv2 = cv2.resize(mask_cv2, (720, 1280))
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            mask = mask_cv2
            ground_truth_mask = (mask > 0)  # turn to T/F mask.
            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask, (1, 1, ground_truth_mask.shape[0], ground_truth_mask.shape[1])))
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            # single rec
            img_recs.append(transformed_data)
            mask_recs.append(gt_binary_mask)

        return img_recs, mask_recs, vid, category, feat_aud, feat_aud

