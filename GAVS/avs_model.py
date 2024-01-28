
import numpy as np
import torch
import torch.nn as nn


class AVSM(nn.Module):
    def __init__(self, model_v, model_t, config):
        super().__init__()
        self.model_v = model_v
        self.model_t = model_t
        self.config = config

        self.device = self.config.get('device')
        self.loss_fn = self.config.get('loss')

    def decode_with_pmp(self, image_embed, audio_embed):
        with torch.no_grad():
            # print(self.model_v.prompt_encoder)
            sparse_embeddings, dense_embeddings = self.model_v.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                audios=audio_embed,
            )

        # mask decoder
        multi_masks = False  # True  # args.multi
        low_res_masks, iou_predictions = self.model_v.mask_decoder(  # lrm: B, N=3-or-1, H_mask=256, W_mask=256
            image_embeddings=image_embed.to(self.device),
            image_pe=self.model_v.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings.to(self.device),
            dense_prompt_embeddings=dense_embeddings.to(self.device),
            multimask_output=multi_masks,
        )

        return low_res_masks, iou_predictions
    
    def add_noise_Gaussian(self, x, m=0, stddev=0.1, noise_strength=1):
        return self.add_gaussian_noise(x=x)

    def add_gaussian_noise(self, x, std=0.1, strngth=0.1):
        std = torch.std(x.view(-1))
        noise = torch.randn_like(x) * std
        # a, b = x.min(), x.max()

        return noise

    def forward(self, batch_data, vid_temporal_mask_flag=None):
        img_recs, mask_recs, vid, category, _, _ = batch_data
        scores = []
        loss_vid = []  # loss for a whole video
        vid_preds = []

        # Decoding
        for _idx, img_rec in enumerate(img_recs):
            input_size = img_rec.get('input_size', None)
            original_size = img_rec.get('original_size', None)
            # image_embed = img_rec.get('image_embed', None)
            audio_embed = img_rec.get('audio', None)
            # print('audio:', audio_embed.shape)  # [1, 128]
            
            # ===== add noise
            if self.training:
                audio_embed = self.add_noise_Gaussian(audio_embed)
            # ============

            # if get img emb from visual encoder:
            if self.config['tune_v'] < 12:
                image_input = img_rec.get('image')  # [1, 3, 1024, 1024]
                image_embed = self.model_v.image_encoder(image_input)

            low_res_masks, iou_predictions = self.decode_with_pmp(image_embed, audio_embed)

            upscaled_masks = self.model_v.postprocess_masks(low_res_masks, input_size, original_size).to(self.device)
            vid_preds.append(upscaled_masks)

            loss_frame = self.loss_fn(upscaled_masks.squeeze(), mask_recs[_idx].squeeze())
            loss_vid.append(loss_frame)
            score = iou_predictions.squeeze()
            scores.append(float(f'{score.item():.2f}'))

        return vid_preds, scores, loss_vid
