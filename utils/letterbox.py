import cv2
import numpy as np
import math

def letterbox_pair(img, mask=None, new_shape=640, color=(114,114,114)):
    h0, w0 = img.shape[:2]
    if isinstance(new_shape, (list, tuple)):
        new_h, new_w = new_shape
    else:
        new_h = new_w = int(new_shape)

    r = min(new_h / h0, new_w / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))   # (w, h)

    dw = new_w - new_unpad[0]
    dh = new_h - new_unpad[1]
    left   = int(math.floor(dw / 2))
    right  = int(math.ceil(dw / 2))
    top    = int(math.floor(dh / 2))
    bottom = int(math.ceil(dh / 2))

    img_rs = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    img_lb = cv2.copyMakeBorder(img_rs, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)

    mask_lb = None
    if mask is not None:
        m_rs = cv2.resize(mask, new_unpad, interpolation=cv2.INTER_NEAREST)
        mask_lb = cv2.copyMakeBorder(m_rs, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=0)
    # 返回四个 pad
    return img_lb, mask_lb, (r, r), (left, right, top, bottom)


def unletterbox_map(prob_lb, orig_hw, ratio, pads):
    """把 letterbox 尺寸上的概率图映射回原图"""
    H0, W0 = orig_hw
    left, right, top, bottom = pads
    Hlb, Wlb = prob_lb.shape[:2]

    # 非对称裁剪
    y0, y1 = top,   Hlb - bottom
    x0, x1 = left,  Wlb - right
    prob_cropped = prob_lb[y0:y1, x0:x1]

    # 直接缩放回原图
    prob = cv2.resize(prob_cropped, (W0, H0), interpolation=cv2.INTER_LINEAR)
    return prob
