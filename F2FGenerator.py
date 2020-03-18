import torch.nn as nn
from model_component import *
import numpy as np

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True, Style_first_layer=True),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
                StyledConvBlock(256, 128, 3, 1, upsample=True, fused=True),  # 128
                StyledConvBlock(128, 64, 3, 1, upsample=True, fused=True)  # 256
                # StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
                # StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1)
                # EqualConv2d(32, 3, 1),
                # EqualConv2d(16, 3, 1),
            ]
        )

    def forward(self, feature, noise, step=6):
        out = np.random.randn(feature.shape[0],512,4,4)
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            out, style_out = conv(out, feature, noise[i])
            feature = style_out
            if i == step:
                out = to_rgb(out)
                return out
        return to_rgb(out)