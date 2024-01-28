
import torch
import torch.nn as nn

class AudioProj(nn.Module):
    """ Project the audio from VGGish[128] to SAM-ViT[256, 64, 64] space. """
    def __init__(self, input_size=128, output_size=256):
        super(AudioProj, self).__init__()

        self.audio_proj = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, output_size),
        )
    def forward(self, aud_feat: torch.Tensor):
        aud_feat = self.audio_proj(aud_feat)
        return aud_feat
