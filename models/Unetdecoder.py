from modules.GBC import *
from modules.DSUB import *

# ============================== CBAM 注意力机制 ==============================
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

# ============================== 大核条纹方向卷积 ==============================
class StripeLargeHybrid(nn.Module):
    def __init__(self, in_ch, out_ch, ks_stripe=(3, 7, 11), k_large=9):
        super().__init__()
        c_each = out_ch // (len(ks_stripe)*2 + 1)  # horiz + vert + 1 large

        # 1) 多尺度条纹（水平 + 垂直）
        self.h_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, c_each, kernel_size=(1, k), padding=(0, k//2), bias=False),
                nn.BatchNorm2d(c_each),
                nn.ReLU(inplace=True)
            )
            for k in ks_stripe
        ])
        self.v_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, c_each, kernel_size=(k, 1), padding=(k//2, 0), bias=False),
                nn.BatchNorm2d(c_each),
                nn.ReLU(inplace=True)
            )
            for k in ks_stripe
        ])

        # 2) 轻量大核：depthwise K×K + 1×1 pointwise
        self.dw_large = nn.Sequential(
            # depthwise: in_ch → in_ch, groups=in_ch
            nn.Conv2d(
                in_ch, in_ch, kernel_size=k_large, padding=k_large//2,
                groups=in_ch, bias=False
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # pointwise: in_ch → c_each
            nn.Conv2d(in_ch, c_each, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_each),
            nn.ReLU(inplace=True)
        )

        total_ch = c_each * (len(ks_stripe)*2 + 1)

        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(total_ch),
            nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(total_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.attn = CBAM(out_ch)  # 可选

    def forward(self, x):
        feats = []
        for m in self.h_convs:
            feats.append(m(x))
        for m in self.v_convs:
            feats.append(m(x))
        feats.append(self.dw_large(x))  # 这里输出的通道数是 c_each

        y = torch.cat(feats, dim=1)      # [B, c_each*(2*len+1), H, W]
        y = self.bn_relu(y)
        y = self.fuse(y)
        y = self.attn(y)
        return y

class MultiScaleStripeConv(nn.Module):
    def __init__(self, in_ch, out_ch, ks_list=(3, 5, 7), use_vert=True, use_horiz=True):
        super().__init__()
        branches = []
        n_branch = len(ks_list) * (int(use_horiz) + int(use_vert))
        c_each = out_ch // n_branch
        self.use_vert = use_vert
        self.use_horiz = use_horiz

        self.cbam = CBAM(out_ch)

        # 水平条纹：1×k，捕获“沿横波方向”的长结构
        if use_horiz:
            self.h_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_ch, c_each, kernel_size=(1, k), padding=(0, k // 2), bias=False),
                    nn.BatchNorm2d(c_each),
                    nn.ReLU(inplace=True)
                ) for k in ks_list
            ])
        else:
            self.h_convs = nn.ModuleList()

        # 垂直条纹：k×1，捕获“跨横波”的变化
        if use_vert:
            self.v_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_ch, c_each, kernel_size=(k, 1), padding=(k // 2, 0), bias=False),
                    nn.BatchNorm2d(c_each),
                    nn.ReLU(inplace=True)
                ) for k in ks_list
            ])
        else:
            self.v_convs = nn.ModuleList()

        self.fuse = nn.Sequential(
            nn.Conv2d(c_each * (len(self.h_convs) + len(self.v_convs)), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = []
        for m in self.h_convs:
            feats.append(m(x))
        for m in self.v_convs:
            feats.append(m(x))
        y = torch.cat(feats, dim=1)
        y = self.cbam(self.fuse(y))
        return y   # 1124修改了这里加入了注意力 best-1124-use是不加这一个注意力的 0.4592


# ============================== 条纹增强卷积模块 ==============================
class StripeEnhancedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, use_large=False):
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

        # self.stripe_conv = MultiScaleStripeConv(in_channels, out_channels // 2)

        # ⭐ 条纹分支：可选 MultiScaleStripeConv 或 StripeLargeHybrid
        if use_large:
            # 条纹 + 大核 hybrid 分支
            self.stripe_conv = StripeLargeHybrid(in_channels, out_channels // 2)
        else:
            # 只用原来的多尺度条纹
            self.stripe_conv = MultiScaleStripeConv(in_channels, out_channels // 2)

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
        out = self.attention(out)
        return out


class EnhancedDoubleConv(nn.Module):
    """
    (Conv3x3-BN-ReLU) × 2 的升级版：
    - 第二层可并联【标准3x3】与【条纹卷积(1x5,5x1)】再融合
    - 可选 CBAM 注意力
    - 可选残差（当 in_ch == out_ch 时）
    """
    def __init__(self,
                 in_ch, out_ch, mid_ch=None,
                 use_stripe=True, use_cbam=True, residual=True):
        super().__init__()
        if mid_ch is None:
            mid_ch = out_ch

        self.use_stripe = use_stripe
        self.use_cbam = use_cbam
        self.residual = residual and (in_ch == out_ch)

        # stage-1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

        # stage-2：标准分支 & 条纹分支
        if use_stripe:
            c_std = out_ch // 2
            c_stripe = out_ch - c_std  # 防止奇数通道丢失
            self.conv2_std = nn.Sequential(
                nn.Conv2d(mid_ch, c_std, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_std),
                nn.ReLU(inplace=True),
            )
            # self.conv2_stripe = SimpleStripeConv(mid_ch, c_stripe)
            self.conv2_stripe = MultiScaleStripeConv(mid_ch, c_stripe)
            self.fuse = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        if self.use_stripe:
            y = torch.cat([self.conv2_std(y), self.conv2_stripe(y)], dim=1)
            y = self.fuse(y)
        else:
            y = self.conv2(y)

        if self.residual:
            y = y + x
        # y = self.cbam(y)
        return y


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch,
                 bilinear=True,
                 use_large=False,   # ⭐ 新增：这一层是否用大核 hybrid
                 **kwargs):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        # ⭐ 这里把 use_large 传给 StripeEnhancedDoubleConv
        self.conv = StripeEnhancedDoubleConv(
            in_ch + skip_ch,
            out_ch,
            mid_channels=out_ch,
            use_large=use_large
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


