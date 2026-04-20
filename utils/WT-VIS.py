import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# ✅ 直接从你自己的 modules/wavelet_cpam.py 引入

class WaveletFrequencyAttention(nn.Module):
    """小波多尺度频率注意力（需要 pywt）"""
    def __init__(self, channel, wavelet='db1', level=1, reduction=16):
        super().__init__()
        if pywt is None:
            raise ImportError("pywt 未安装：WaveletFrequencyAttention 需要 PyWavelets。")

        self.channel = channel
        self.wavelet = wavelet
        self.level = level
        self.num_subbands = 3 * level + 1

        filters = self.get_wavelet_filters()
        for name, filt in filters.items():
            self.register_buffer(name, filt)

        self.freq_fc = nn.Sequential(
            nn.Linear(channel * self.num_subbands, max(channel // reduction, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 8), channel, bias=False),
            nn.Sigmoid()
        )

    def get_wavelet_filters(self):
        wavelet = pywt.Wavelet(self.wavelet)
        dec_lo = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        dec_hi = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32)

        ll = dec_lo.view(1, -1) * dec_lo.view(-1, 1)
        lh = dec_lo.view(1, -1) * dec_hi.view(-1, 1)
        hl = dec_hi.view(1, -1) * dec_lo.view(-1, 1)
        hh = dec_hi.view(1, -1) * dec_hi.view(-1, 1)

        return {
            'll_filter': ll.unsqueeze(0).unsqueeze(0),
            'lh_filter': lh.unsqueeze(0).unsqueeze(0),
            'hl_filter': hl.unsqueeze(0).unsqueeze(0),
            'hh_filter': hh.unsqueeze(0).unsqueeze(0),
        }

    def wavelet_decompose(self, x):
        b, c, h, w = x.shape
        subbands = []
        current = x
        k = self.ll_filter.shape[-1]
        pad = k // 2

        for _ in range(self.level):
            c = current.shape[1]
            ll = F.conv2d(current, self.ll_filter.repeat(c, 1, 1, 1), stride=2, padding=pad, groups=c)
            lh = F.conv2d(current, self.lh_filter.repeat(c, 1, 1, 1), stride=2, padding=pad, groups=c)
            hl = F.conv2d(current, self.hl_filter.repeat(c, 1, 1, 1), stride=2, padding=pad, groups=c)
            hh = F.conv2d(current, self.hh_filter.repeat(c, 1, 1, 1), stride=2, padding=pad, groups=c)

            subbands.extend([lh, hl, hh])
            current = ll

        subbands.append(current)
        return subbands

    def forward(self, x):
        b, c, _, _ = x.shape
        subbands = self.wavelet_decompose(x)

        pooled = []
        for sb in subbands:
            pooled.append(torch.mean(sb, dim=(2, 3)))  # (B,C)

        feat = torch.cat(pooled, dim=1)               # (B, C*num_subbands)
        w = self.freq_fc(feat).view(b, c, 1, 1)       # (B,C,1,1)
        return w



def _to_tensor_image(img_path, to_gray=True):
    """
    读取图像 -> torch tensor: (1,C,H,W), float32
    to_gray=True: 强制灰度 (C=1)，更直观
    """
    if to_gray:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        img = img.astype(np.float32) / 255.0
        x = torch.from_numpy(img)[None, None, :, :]  # (1,1,H,W)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)[None, :, :, :]  # (1,3,H,W)
    return x


def _to_showable(arr):
    """
    arr: (h,w) numpy float
    做 min-max 归一化到 0~255 方便显示/保存
    """
    a = arr.copy()
    a = a - a.min()
    if a.max() > 1e-8:
        a = a / a.max()
    a = (a * 255.0).clip(0, 255).astype(np.uint8)
    return a


@torch.no_grad()
def visualize_wavelet_subbands(img_path, wavelet='haar', out_dir='./wavelet_vis',
                               device='cpu', to_gray=True):
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读图 -> (1,C,H,W)
    x = _to_tensor_image(img_path, to_gray=to_gray).to(device)

    # 2) 用你的 WaveletFrequencyAttention 做 level=1 分解
    #    reduction 只是给 freq_fc 用的，这里只用 wavelet_decompose，不影响子带输出
    wfa = WaveletFrequencyAttention(channel=x.shape[1], wavelet=wavelet, level=1, reduction=16).to(device)
    wfa.eval()

    # 3) 拿到子带。你实现返回顺序： [LH, HL, HH, LL]
    subbands = wfa.wavelet_decompose(x)
    lh, hl, hh, ll = subbands[0], subbands[1], subbands[2], subbands[3]

    # 4) 为了可视化：把 (1,C,h,w) -> (h,w)
    #    若 C>1（RGB），这里用通道均值更直观
    def pack(sb):
        sb2 = sb[0].mean(dim=0).detach().cpu().numpy()  # (h,w)
        return sb2

    LH = pack(lh)
    HL = pack(hl)
    HH = pack(hh)
    LL = pack(ll)

    # 5) 归一化成可显示的 uint8
    vis = {
        "LL": _to_showable(LL),
        "LH": _to_showable(LH),
        "HL": _to_showable(HL),
        "HH": _to_showable(HH),
    }

    # 6) 保存图像（灰度png）
    for k, v in vis.items():
        save_path = os.path.join(out_dir, f"{k}_{wavelet}_level1.png")
        cv2.imwrite(save_path, v)
        print("saved:", save_path)

    # 7) 画出来（2x2）
    plt.figure(figsize=(10, 8))
    for i, k in enumerate(["LL", "LH", "HL", "HH"], 1):
        plt.subplot(2, 2, i)
        plt.imshow(vis[k], cmap='gray')
        plt.title(k)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    img_path = "/home/lab/libr/obb-RetinaNet/wave_dataset/JPEGImages/20455.jpg"  # 改成你的图像路径
    visualize_wavelet_subbands(
        img_path=img_path,
        wavelet="haar",
        out_dir="/home/lab/libr/obb-RetinaNet/outputs/integrated",
        device="cuda" if torch.cuda.is_available() else "cpu",
        to_gray=True  # ✅ 建议先用灰度看得最清楚
    )
