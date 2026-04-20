import typing as t
import torch
import torch.nn as nn

class ShareableAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        gate_layer: str = 'sigmoid',
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, '输入特征的维度应能被4整除。'
        self.group_chans = group_chans = self.dim // 4

        self.local_dwc = nn.Conv1d(
            group_chans, group_chans,
            kernel_size=group_kernel_sizes[0],
            padding=group_kernel_sizes[0] // 2,
            groups=group_chans
        )
        self.global_dwc_s = nn.Conv1d(
            group_chans, group_chans,
            kernel_size=group_kernel_sizes[1],
            padding=group_kernel_sizes[1] // 2,
            groups=group_chans
        )
        self.global_dwc_m = nn.Conv1d(
            group_chans, group_chans,
            kernel_size=group_kernel_sizes[2],
            padding=group_kernel_sizes[2] // 2,
            groups=group_chans
        )
        self.global_dwc_l = nn.Conv1d(
            group_chans, group_chans,
            kernel_size=group_kernel_sizes[3],
            padding=group_kernel_sizes[3] // 2,
            groups=group_chans
        )

        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()

        # 注意：这里 GroupNorm 的 num_groups=4，要求 dim 能被 4 整除（你上面已 assert）
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h_, w_ = x.size()

        # H 方向：对 W 求均值 -> (B,C,H)
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)

        # W 方向：对 H 求均值 -> (B,C,W)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # H 注意力
        x_h_attn = torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)
        x_h_attn = self.sa_gate(self.norm_h(x_h_attn)).view(b, c, h_, 1)

        # W 注意力
        x_w_attn = torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l),
        ), dim=1)
        x_w_attn = self.sa_gate(self.norm_w(x_w_attn)).view(b, c, 1, w_)

        return x * x_h_attn * x_w_attn


class DSConvAttn(nn.Module):
    """
    Depthwise Separable Conv Attention
    - DWConv(3x3) 做局部细化
    - SE 做通道重标定
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
            nn.Conv2d(channels, mid, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.act(self.bn(self.pw(self.dw(x))))
        w = self.se(y)
        return y * w


class MSAM(nn.Module):
    def __init__(self, channels_high, channels_output=256, divchannel=4,
                 dsca_reduction=16, use_residual=False):
        super().__init__()
        self.conv_high = nn.Sequential(
            nn.Conv2d(channels_high, channels_output, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels_output),
            nn.ReLU(inplace=True)
        )

        self.fc1 = nn.Linear(channels_output, channels_output // divchannel)
        self.fc2 = nn.Linear(channels_output, channels_output // divchannel)
        self.softmax = nn.Softmax(dim=-1)

        # 原作者稳定项（从 0 学起）
        self.scale = nn.Parameter(torch.zeros(1))
        self.output_channels = channels_output

        # 可分离卷积注意力细化
        self.dsca = DSConvAttn(channels_output, reduction=dsca_reduction, k=3)

        self.use_residual = use_residual

    def forward(self, fms_low, fms_high):
        b, c1, h1, w1 = fms_low.shape
        if c1 != self.output_channels:
            raise ValueError(f"fms_low channels={c1} != output_channels={self.output_channels}")

        # (B, HW_low, C)
        feat_low = fms_low.permute(0, 2, 3, 1).contiguous().view(b, -1, self.output_channels)

        # 高层先 1x1 对齐到 output_channels，然后展开
        high = self.conv_high(fms_high)
        feat_high = high.permute(0, 2, 3, 1).contiguous().view(b, -1, self.output_channels)

        # Q: low, K: high
        mid_low  = self.fc1(feat_low)                       # (B, HW_low, C/div)
        mid_high = self.fc2(feat_high).permute(0, 2, 1)     # (B, C/div, HW_high)

        # attention: (B, HW_low, HW_high)
        energy = torch.bmm(mid_low, mid_high)
        attention = self.softmax(energy)

        # V: high -> 回写到 low 空间
        mid = torch.bmm(
            feat_high.permute(0, 2, 1),                     # (B, C, HW_high)
            attention.permute(0, 2, 1)                      # (B, HW_high, HW_low)
        ).view(b, self.output_channels, h1, w1)

        # DSConv 注意力细化
        mid_refine = self.dsca(mid)

        out = self.scale * mid_refine
        if self.use_residual:
            out = out + fms_low
        return out
