import torch.nn as nn
import torch
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, 8, kernel_size=4, stride=2, padding=1) # 128
        self.conv2 = nn.Conv2d(8, 16, 4, 2, 1) # 64
        self.conv3 = nn.Conv2d(16, 32, 4, 2, 1) # 32
        self.conv4 = nn.Conv2d(32, 64, 4, 2, 1) # 16
        self.conv5 = nn.Conv2d(64, 64, 4, 2, 1) # 8

        self.b1 = nn.BatchNorm2d(8)
        self.b2 = nn.BatchNorm2d(16)
        self.b3 = nn.BatchNorm2d(32)
        self.b4 = nn.BatchNorm2d(64)
        self.b5 = nn.BatchNorm2d(64)

        self.linear = nn.Linear(64*8*8, 64)

    def forward(self, inputs):
        out = F.relu(self.b1(self.conv1(inputs)))
        out = F.relu(self.b2(self.conv2(out)))
        out = F.relu(self.b3(self.conv3(out)))
        out = F.relu(self.b4(self.conv4(out)))
        out = F.relu(self.b5(self.conv5(out))).view(-1, 64*8*8)
        out = torch.sigmoid(self.linear(out))
        return out

