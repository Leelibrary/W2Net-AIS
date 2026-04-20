import torch
import torch.nn as nn
import torch.nn.functional as F


class StripeEnhancedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.standard_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.stripe_conv = SimpleStripeConv(in_channels, out_channels // 2)

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = CBAM(out_channels)

    def forward(self, x):
        standard_out = self.standard_conv(x)
        stripe_out = self.stripe_conv(x)
        fused = torch.cat([standard_out, stripe_out], dim=1)
        out = self.final_conv(fused)
        # out = self.attention(out)
        return out


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        ca = self.sigmoid_channel(avg_out + max_out)
        x = x * ca

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.sigmoid_spatial(self.conv_spatial(sa_input))
        x = x * sa

        return x

# ------------------ GBC 中的 BottConv ------------------
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size,
                 stride=1, padding=0, bias=False):
        super().__init__()
        self.pw1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=bias)
        self.dw  = nn.Conv2d(mid_channels, mid_channels, kernel_size,
                             stride=stride, padding=padding,
                             groups=mid_channels, bias=False)
        self.pw2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.pw1(x)
        x = self.dw(x)
        x = self.pw2(x)
        return x


class BottConv_Stripe(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels,
                 kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # 1x1 降通道
        self.reduce = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        # 条纹卷积，提方向信息
        self.stripe = SimpleStripeConv(mid_channels, mid_channels)
        # 1x1 升通道
        self.expand = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.reduce(x)
        x = self.stripe(x)
        x = self.expand(x)
        return x


# -------------------- Simple Stripe Conv --------------------
class SimpleStripeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_h = nn.Conv2d(in_channels, out_channels//2, (1,5), padding=(0,2), bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels//2, (5,1), padding=(2,0), bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv_h(x)
        v = self.conv_v(x)
        out = torch.cat([h, v], dim=1)
        return self.act(self.bn(out))


def get_norm_layer(norm_type, channels, num_groups):
    if norm_type == 'GN':
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels)
    else:
        return nn.InstanceNorm3d(channels)


# ------------------ ⭐【核心】GBC-StripeConv ------------------
class GBC(nn.Module):
    def __init__(self, in_channels, norm_type='GN', att_type='none'):
        super(GBC, self).__init__()

        self.block1 = nn.Sequential(
            BottConv_Stripe(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            BottConv_Stripe(in_channels, in_channels, in_channels // 8, 3, 1, 1),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
        )

        self.stripe_block = StripeEnhancedDoubleConv(
            in_channels=in_channels,
            out_channels=in_channels,  # 保持通道不变，方便后面相乘
            mid_channels=None  # 不写就默认=out_channels
        )

        self.block3 = nn.Sequential(
            BottConv_Stripe(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
        )

        self.block4 = nn.Sequential(
            BottConv_Stripe(in_channels, in_channels, in_channels // 8, 1, 1, 0),
            get_norm_layer(norm_type, in_channels, 16),
            nn.ReLU(inplace=True),
        )

        self.gate_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gate_bn = nn.BatchNorm2d(in_channels)

        self.att_type = att_type.lower()
        if self.att_type == 'cbam':
            self.att = CBAM(in_channels)
        else:
            self.att = nn.Identity()

    # def forward(self, x):  # 1121
    #     residual = x
    #
    #     # x1 = self.block1(x)
    #     # x1 = self.block2(x1)
    #
    #     x1 = self.stripe_block(x)
    #
    #     x2 = self.block3(x)
    #
    #     x = x1 * x2
    #     x = self.block4(x)
    #
    #     # x = self.att(x)
    #
    #     return x + residual
    def forward(self, x):
        residual = x

        # 分支1：Stripe + 标准卷积增强
        x1 = self.stripe_block(x)    # (B,C,H,W)

        # 分支2：轻量 BottConv_Stripe 提取一个调制信号
        x2 = self.block3(x)          # (B,C,H,W)

        # 用 x2 生成一个 [0,1] 的 gate，去调制 x1，而不是直接相乘
        gate = torch.sigmoid(self.gate_bn(self.gate_conv(x2)))  # (B,C,H,W)

        # ★ 关键：x2 只作为“门控”，不会直接参与乘法放大/压小特征
        x = x1 * gate

        # x = self.block4(x)

        # x = self.att(x)

        return x + residual



class GBCStripeConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='GN', att_type='none'):
        super().__init__()
        # 先调整到 out_channels，再喂给 GBC
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

        self.gbc = GBC(out_channels, norm_type=norm_type)

    def forward(self, x):
        x = self.proj(x)
        x = self.gbc(x)
        return x

# 1120

