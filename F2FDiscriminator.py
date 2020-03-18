from model_component import *

class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                # ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
                # ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
                ConvBlock(3, 128, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(256, 512, 3, 1, downsample=True),  # 32
                ConvBlock(512, 512, 3, 1, downsample=True),  # 16
                ConvBlock(512, 512, 3, 1, downsample=True),  # 8
                ConvBlock(512, 512, 3, 1, downsample=True),  # 4
                # ConvBlock(513, 512, 3, 1, 4, 0)
            ]
        )

        def make_from_rgb(out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(3, out_channel, 1), nn.LeakyReLU(0.2))
            else:
                return EqualConv2d(3, out_channel, 1)

        self.from_rgb = nn.ModuleList(
            [
                # make_from_rgb(16),
                # make_from_rgb(32),
                make_from_rgb(64),
                make_from_rgb(128),
                make_from_rgb(256),
                make_from_rgb(512),
                make_from_rgb(512),
                make_from_rgb(512)
                # make_from_rgb(512)
            ]
        )

        self.n_layer = len(self.progression)

    def forward(self, input, step=6):
        out = input
        for i in range(self.n_layer):
            out = self.progression[i](out)
            if i == step:
                out = self.from_rgb[i](out)
                out = torch.sigmoid(out)
                return out.squeeze()
            else:
                continue
        out = torch.sigmoid(out)
        out = out.squeeze()
        return out
