
import torch
import torch.nn as nn

class VisualProj(nn.Module):
    """ Project the audio from VGGish[128] to SAM-ViT[256, 64, 64] space. """
    def __init__(self, input_size=256, output_size=256):
        super(VisualProj, self).__init__()

        self.visual_proj = nn.Sequential(  # 对比学习
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )
    def forward(self, img_feat: torch.Tensor):
        batch_size = img_feat.size(0)  # [20, 128]
        img_feat_proj = self.visual_proj(img_feat)  # [20, 256]
        return img_feat + img_feat_proj
