import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    对应论文里的 Convolution Block (CB)：
    CB = LR(BN(Conv1x1(LR(BN(DWConv3x3(x))))))
    使用 depthwise 3x3 + pointwise 1x1，带 BN + LeakyReLU
    """
    def __init__(self, channels, negative_slope=0.01):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(channels)

        self.pw_conv = nn.Conv2d(
            channels, channels,
            kernel_size=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(channels)

        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.act(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.act(x)

        return x

class DSUB(nn.Module):
    """
    Depth-to-Space Upsampling Block (DSUB)

    作用：用 Depth-to-Space (PixelShuffle) 做上采样，尽量保留 encoder 累积的细节，
    然后再用卷积块细化特征。

    结构对应：
    DSUB = CB( ReLU( Conv3x3( D2S( ReLU( Conv3x3(x) ) ) ) ) )

    参数：
        in_channels   : 输入通道 C_in
        out_channels  : 输出通道 C_out
        scale         : 上采样倍数（一般=2）
    """
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale

        # 1) 预卷积：把通道提到 C_in * scale^2，供 PixelShuffle 重排
        # 文中 F = C_prev * 2^d，d=2 时就是 4*C_prev :contentReference[oaicite:2]{index=2}
        self.pre_conv = nn.Conv2d(
            in_channels,
            in_channels * (scale ** 2),
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.pre_bn = nn.BatchNorm2d(in_channels * (scale ** 2))

        # 2) Depth-to-Space 上采样
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # 3) 上采样后的 3x3 卷积，调整到 out_channels
        self.post_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.post_bn = nn.BatchNorm2d(out_channels)

        # 4) 最后一个 Convolution Block（depthwise+1x1）
        self.cb = ConvBlock(out_channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C_in, H, W)

        # Conv3x3 + ReLU
        x = self.pre_conv(x)
        x = self.pre_bn(x)
        x = self.act(x)  # (B, C_in*scale^2, H, W)

        # Depth-to-Space / PixelShuffle: (B, C_in, H*scale, W*scale)
        x = self.pixel_shuffle(x)

        # Conv3x3 + ReLU
        x = self.post_conv(x)
        x = self.post_bn(x)
        x = self.act(x)  # (B, C_out, H*scale, W*scale)

        # Convolution Block 细化
        x = self.cb(x)

        return x


class EUB(nn.Module):
    """
    Effective Upsampling Block (EUB)

    公式：EUB = CB( BI( CB(x) ) )
    - 先用 CB 做一次特征净化（去掉前面 stage 的噪声/无效特征）
    - 再用双线性插值上采样
    - 上采样后再用 CB 做一次细化

    这里默认只做通道保持：C -> C，方便替换 UpBlock 里的上采样模块。
    """
    def __init__(self, channels, scale=2, negative_slope=0.01):
        super().__init__()
        self.cb1 = ConvBlock(channels, negative_slope=negative_slope)
        self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
        self.cb2 = ConvBlock(channels, negative_slope=negative_slope)

    def forward(self, x):
        # 第一次卷积块：去噪 + 提纯
        x = self.cb1(x)
        # 双线性插值上采样
        x = self.up(x)
        # 上采样后的细化
        x = self.cb2(x)
        return x
