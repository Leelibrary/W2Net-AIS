import torch
import torch.nn as nn

# 例：简单版 SPPF
class SPPF(nn.Module):
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        self.cv1 = nn.Conv2d(c_in, c_out, 1, 1, 0)
        self.m   = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
        self.cv2 = nn.Conv2d(c_out * 4, c_out, 1, 1, 0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))
