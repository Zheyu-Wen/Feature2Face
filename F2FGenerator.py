import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, input_width):
        super().__init__()
        self.input_width = input_width
        self.linear = nn.Linear(4096, 4096) # 64
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1) # 32
        self.conv2 = nn.Conv2d(8, 32, kernel_size=4, stride=2, padding=1)  # 16
        self.convT3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32
        self.convT4 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 64
        self.convT5 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1) # 128
        self.convT6 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1) # 256

        self.b1 = nn.BatchNorm2d(8)
        self.b2 = nn.BatchNorm2d(32)
        self.b3 = nn.BatchNorm2d(32)
        self.b4 = nn.BatchNorm2d(16)
        self.b5 = nn.BatchNorm2d(8)


        self.convN1 = nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=1)  # for noise
        self.convN2 = nn.Conv2d(8, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)



    def forward(self, inputs, noise):
        out = F.relu(self.linear(inputs)).view(-1, 1, self.input_width, self.input_width)
        out = F.relu(self.b1(self.conv1(out)))
        out = F.relu(self.b2(self.conv2(out)))
        outN = F.relu(self.bn1(self.convN1(noise)))
        outN = F.relu((self.bn2(self.convN2(outN))))
        out = torch.cat([out, outN], 1)
        out = F.relu(self.b3(self.convT3(out)))
        out = F.relu(self.b4(self.convT4(out)))
        out = F.relu(self.b5(self.convT5(out)))
        out = torch.tanh(self.convT6(out))
        return out

