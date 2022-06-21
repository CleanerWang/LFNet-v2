import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)

    def forward(self, x):
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x+y)

class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config

        #-------LFNetv2-------------#

        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.res1 = ResidualBlock(64)

        self.conv2 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.norm2 = nn.InstanceNorm2d(128, affine=True)
        self.res2 = ResidualBlock(128)

        self.conv3 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.norm3 = nn.InstanceNorm2d(256, affine=True)
        self.res3 = ResidualBlock(256)

        self.conv4 = nn.Conv2d(256, 256, 3, 2, 1, bias=False)
        self.norm4 = nn.InstanceNorm2d(256, affine=True)
        self.res4 = ResidualBlock(256)

        self.conv5 = nn.Conv2d(256, 256, 3, 2, 1, bias=False)
        self.norm5 = nn.InstanceNorm2d(256, affine=True)
        self.res5 = ResidualBlock(256)

        self.conv3_1 = nn.Conv2d(256, 7, 3, 1, 1, bias=False)
        self.conv3_2 = nn.Conv2d(256, 7, 3, 2, 1, bias=False)
        self.conv3_3 = nn.Conv2d(256, 7, 6, 4, 1, bias=False)

        self.conv4_1 = nn.ConvTranspose2d(256, 7, 2, 2, 0)
        self.conv4_2 = nn.Conv2d(256, 7, 3, 1, 1, bias=False)
        self.conv4_3 = nn.Conv2d(256, 7, 3, 2, 1, bias=False)

        self.conv5_1 = nn.ConvTranspose2d(256, 7, 4, 4, 0)
        self.conv5_2 = nn.ConvTranspose2d(256, 7, 2, 2, 0)
        self.conv5_3 = nn.Conv2d(256, 7, 3, 1, 1, bias=False)



    def forward(self, x):

        x1 = F.relu(self.norm1(self.conv1(x)))
        x1 = self.res1(x1)
        x1 = self.res1(x1)
        x1 = self.res1(x1)
        x1 = self.res1(x1)
        x1 = self.res1(x1)

        x2 = F.relu(self.norm2(self.conv2(x1)))
        x2 = self.res2(x2)
        x2 = self.res2(x2)
        x2 = self.res2(x2)
        x2 = self.res2(x2)
        x2 = self.res2(x2)

        x3 = F.relu(self.norm3(self.conv3(x2)))
        x3 = self.res3(x3)
        x3 = self.res3(x3)
        x3 = self.res3(x3)
        x3 = self.res3(x3)
        x3 = self.res3(x3)

        x4 = F.relu(self.norm4(self.conv4(x3)))
        x4 = self.res4(x4)
        x4 = self.res4(x4)
        x4 = self.res4(x4)
        x4 = self.res4(x4)
        x4 = self.res4(x4)

        x5 = F.relu(self.norm5(self.conv5(x4)))
        x5 = self.res5(x5)
        x5 = self.res5(x5)
        x5 = self.res5(x5)
        x5 = self.res5(x5)
        x5 = self.res5(x5)


        x3_1 = self.conv3_1(x3)
        x3_2 = self.conv3_2(x3)
        x3_3 = self.conv3_3(x3)

        x4_1 = self.conv4_1(x4)
        x4_2 = self.conv4_2(x4)
        x4_3 = self.conv4_3(x4)

        x5_1 = self.conv5_1(x5)
        x5_2 = self.conv5_2(x5)
        x5_3 = self.conv5_3(x5)


        out0 = torch.cat((x3_3, torch.cat((x4_3, x5_3), 1)), 1)
        out1 = torch.cat((x3_2, torch.cat((x4_2, x5_2), 1)), 1)
        out2 = torch.cat((x3_1, torch.cat((x4_1, x5_1), 1)), 1)
        return out0, out1, out2

