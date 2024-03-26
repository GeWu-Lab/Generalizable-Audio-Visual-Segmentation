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

# logger = log_agent('audio_recs.log')

import pickle as pkl

def load_image_in_PIL_to_Tensor(path, mode='RGB', transform=None):
    img_PIL = Image.open(path).convert(mode)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL

class AVS(Dataset):
    def __init__(self, metafile='meta_v3_seen_train', feature_dir='', model=None, device=None, audio_from=None):
        # metadata: train/test/val
        self.device = device
        self.model = model
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

        meta_path = f'segment_anything/dataset/v3/{metafile}.csv'  # one-shot: metafile='v3_1_shot/train.csv'
        self.metadata = pd.read_csv(meta_path, header=0)

        self.audio = None
        self.images = None

        self.mask_transform = transforms.Compose([transforms.ToTensor()])

        # self.logger = log_agent('dataset.log')

        self.data_base_path = '../../data/AVS/'
        self.feat_path = f'../segment_anything/feature_extract'

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid, category, ver = df_one_video['uid'], df_one_video['a_obj'], df_one_video['label']  # uid for data id.
        self.data_path = f'../../data/AVS/{ver}'
        FN = 10 if ver == 'v2' else 5

        img_recs = []
        mask_recs = []

        feat_aud_p = f'{self.feat_path}/{self.ver}_vggish_embs/{vid}.npy'
        feat_aud = torch.from_numpy(np.load(feat_aud_p)).to(self.device).squeeze().detach()

        for _idx in range(FN):  # set frame_num as the batch_size
            path_frame = f'{self.data_path}/{vid}/frames/{_idx}.jpg'  # image

            # cheange to your own extracted feature path
            # if you would like to extract feature during the training, please comment this block and return null variable.
            feat_img_p = f'{self.feat_path}/{self.ver}_img_embed/{vid}_f{_idx}.pth'  # image feature
            image_embed = torch.load(feat_img_p).squeeze().to(self.device)

            # data
            transformed_data = defaultdict(dict)
            image = cv2.imread(path_frame)
            # image = cv2.resize(image, (720, 1280))

            input_image = self.transform.apply_image(image)

            input_image_torch = torch.as_tensor(input_image, device=self.device)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            # prepare for input
            input_image = self.model.preprocess(transformed_image)
            # print(image.shape)
            original_image_size = (image.shape[0], image.shape[1])  # H x W
            input_size = tuple(transformed_image.shape[-2:])

            # embedding
            audio_embed = feat_aud[_idx].squeeze().to(self.device)

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
            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_mask, (1, 1, ground_truth_mask.shape[0], ground_truth_mask.shape[1]))).to(self.device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)

            # single rec
            img_recs.append(transformed_data)
            mask_recs.append(gt_binary_mask)

        return img_recs, mask_recs, vid, category, feat_aud, feat_aud

