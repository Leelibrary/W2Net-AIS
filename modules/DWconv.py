import torch
import torch.nn as nn
import torch.nn.functional as F


class DSConvAttn(nn.Module):
    """
    Depthwise Separable Conv Attention
    - depthwise 3x3：空间细化（对细结构更友好）
    - pointwise 1x1：通道混合
    - SE gating：通道注意力
    """
    def __init__(self, channels, reduction=16, k=3):
        super().__init__()
        pad = k // 2

        self.dw = nn.Conv2d(channels, channels, kernel_size=k, padding=pad, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.act(self.bn(self.pw(self.dw(x))))
        w = self.se(y)
        return y * w
