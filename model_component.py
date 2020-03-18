import torch
import torchvision
import torch.nn.functional as F
from math import sqrt
import torch.nn as nn


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out

class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out

class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = conv

    def forward(self, input):
        return self.conv(input)

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class FeatureStyle(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, first_layer=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.convResize1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.convResize2 = nn.Conv2d(16, 64, kernel_size=4, stride=2, padding=1)
        self.convResize3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.convResize4 = nn.Conv2d(128, 512, kernel_size=4, stride=2, padding=1)

        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=self.kernel_size, stride=2,
                                       padding=self.padding)
        self.first_layer = first_layer

    def forward(self, inputs):
        resized_feature = inputs
        if self.first_layer:
            batch_size = inputs.shape[0]
            resized_feature = torch.reshape(resized_feature, (batch_size, 1, 64, 64))
            resized_feature = self.convResize1(resized_feature)
            resized_feature = self.convResize2(resized_feature)
            resized_feature = self.convResize3(resized_feature)
            resized_feature = self.convResize4(resized_feature)
        else:
            resized_feature = self.conv(resized_feature)
        return resized_feature

# the convolution block of Generator
class StyledConvBlock(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, initial=False, upsample=False, fused=False, Style_first_layer=False):
    super().__init__()
    if initial:
        self.conv1 = ConstantInput(in_channel)
    else:
        if upsample:
            if fused:
                self.conv1 = nn.Sequential(
                    FusedUpsample(
                        in_channel, out_channel, kernel_size, padding=padding
                    )
                    # Blur(out_channel),
                )

            else:
                self.conv1 = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    EqualConv2d(
                        in_channel, out_channel, kernel_size, padding=padding
                    )
                    # Blur(out_channel),
                )

        else:
            self.conv1 = EqualConv2d(
                in_channel, out_channel, kernel_size, padding=padding
            )

    self.noise1 = NoiseInjection(out_channel)
    self.adain1 = FeatureStyle(in_channel, out_channel, 4, 1, Style_first_layer)
    self.lrelu1 = nn.LeakyReLU(0.2)

    self.conv2 = EqualConv2d(out_channel*2, out_channel, kernel_size, padding=padding)
    # self.noise2 = NoiseInjection(out_channel)
    # self.adain2 = FeatureStyle(out_channel, out_channel)
    self.lrelu2 = nn.LeakyReLU(0.2)

  def forward(self, inputs, style, noise):
      out = self.conv1(inputs)
      out = self.noise1(out, noise)
      out = self.lrelu1(out)
      style_out = self.adain1(style)
      concat_out = torch.cat([out, style_out], 1)

      out = self.conv2(concat_out)
      # out = self.noise2(out, noise)
      out = self.lrelu2(out)

      return out, style_out

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)

        return out