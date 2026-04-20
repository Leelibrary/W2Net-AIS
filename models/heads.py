import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Unetdecoder import EnhancedDoubleConv, UpBlock, StripeEnhancedDoubleConv

class CLSHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_classes):
        super(CLSHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.head = nn.Conv2d(feat_channels, num_anchors*num_classes, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(-math.log((1.0 - prior) / prior))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = torch.sigmoid(self.head(x))
        x = x.permute(0, 2, 3, 1)
        n, w, h, c = x.shape
        x = x.reshape(n, w, h, self.num_anchors, self.num_classes)
        return x.reshape(x.shape[0], -1, self.num_classes)


class REGHead(nn.Module):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_stacked,
                 num_anchors,
                 num_regress):
        super(REGHead, self).__init__()
        assert num_stacked >= 1, ''
        self.num_anchors = num_anchors
        self.num_regress = num_regress
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            chns = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(chns, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.head = nn.Conv2d(feat_channels, num_anchors*num_regress, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.head.weight.data.fill_(0)
        self.head.bias.data.fill_(0)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.head(x)
        x = x.permute(0, 2, 3, 1)
        return x.reshape(x.shape[0], -1, self.num_regress)


class SEGHead(nn.Module):

    def __init__(self, in_channels=256, feat_channels=256, num_stacked=3, out_channels=1):
        super().__init__()
        assert num_stacked >= 1
        self.convs = nn.ModuleList()
        for i in range(num_stacked):
            ch = in_channels if i == 0 else feat_channels
            self.convs.append(nn.Conv2d(ch, feat_channels, 3, 1, 1))
            self.convs.append(nn.ReLU(inplace=True))
        self.pred = nn.Conv2d(feat_channels, out_channels, 1, 1, 0)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, P3, P4, P5, out_hw):
        # 对齐尺度后相加

        # 用 P3 的空间尺寸作为基准，避免奇偶数导致的 off-by-one
        h3, w3 = P3.shape[-2:]
        x3 = P3
        x4 = F.interpolate(P4, size=(h3, w3), mode='bilinear', align_corners=False)
        x5 = F.interpolate(P5, size=(h3, w3), mode='bilinear', align_corners=False)
        x  = x3 + x4 + x5
        for layer in self.convs:
            x = layer(x)
        x  = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
        return self.pred(x)  # logits，未过sigmoid


class FPNUNet5SegHead(nn.Module):
    def __init__(self, fpn_channels=256, out_channels=1, bilinear=True):
        super().__init__()
        C = fpn_channels

        # ⭐ 这里认为：
        # up65 / up54 / up43 对应的就是 P5→P4→P3 这一段（中高层）
        # → 用 large hybrid
        self.up65 = UpBlock(C, C, C, bilinear, use_large=False)   # P5 + P4
        self.up54 = UpBlock(C, C, C, bilinear, use_large=True)   # → P4 + P3
        self.up43 = UpBlock(C, C, C, bilinear, use_large=True)   # → P3 + P2

        # ⭐ 最后贴近输出的这一层（P2 + P1），保持轻一点，用纯条纹/普通卷积即可
        self.up32 = UpBlock(C, C, C, bilinear, use_large=True)  # → P2 + P1

        self.head = nn.Sequential(
            EnhancedDoubleConv(C, C, use_stripe=False, use_cbam=False, residual=False),
            nn.Conv2d(C, out_channels, kernel_size=1)
        )

    def forward(self, P1, P2, P3, P4, P5, out_hw):
        x = P5
        x = self.up65(x, P4)  # hybrid
        x = self.up54(x, P3)  # hybrid
        x = self.up43(x, P2)  # hybrid
        x = self.up32(x, P1)  # 只用原条纹/普通 conv
        x = self.head(x)
        x = F.interpolate(x, size=out_hw, mode='bilinear', align_corners=False)
        return x
