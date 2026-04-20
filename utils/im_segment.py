import numpy as np
import torch.nn.functional as F
import cv2

def im_segment(model, im_rgb, target_size=640):
    """
    输入:
      - model: 你的 RetinaNet（eval 模式）
      - im_rgb: numpy uint8, (H,W,3), RGB
      - target_size: 与检测一致的短边尺度 (int 或 [int])
    返回:
      - seg_logits_up: torch.Tensor, (1,1,H,W) —— 已经上采样回原图大小
    """
    model_was_training = model.training
    model.eval()

    H0, W0 = im_rgb.shape[:2]
    if isinstance(target_size, (list, tuple)):
        target_size = target_size[0]

    # 简单等比缩放到短边 target_size，并 pad 到 32 的倍数（与检测一致即可）
    # 你也可以直接复用 utils.utils.Rescale + Normailize 的实现
    import math
    import torch
    im = im_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    # 等比缩放
    r = float(target_size) / min(H0, W0)
    h, w = int(round(H0 * r)), int(round(W0 * r))
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
    # pad 到 32 倍数
    ph, pw = int(math.ceil(h / 32.0) * 32 - h), int(math.ceil(w / 32.0) * 32 - w)
    im = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode='constant', constant_values=0)
    # HWC -> CHW -> NCHW
    im_t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).contiguous().float()
    im_t = im_t.cuda() if next(model.parameters()).is_cuda else im_t

    # 前向：尝试两种调用方式（二选一）
    with torch.no_grad():
        # 方式1：模型 forward 支持 return_seg=True
        try:
            outs = model(im_t, return_seg=True)
            # 兼容字典/元组
            if isinstance(outs, dict) and 'seg_logits' in outs:
                seg_logits_s = outs['seg_logits']       # (1,1,h',w')
            else:
                seg_logits_s = outs  # 假定直接返回 logits
        except TypeError:
            # 方式2：有独立分割前向
            if hasattr(model, 'seg_forward'):
                seg_logits_s = model.seg_forward(im_t)  # (1,1,h',w')
            else:
                raise RuntimeError('Model does not expose segmentation inference. '
                                   'Please implement forward(..., return_seg=True) or seg_forward(x).')

        # 上采样回原图大小 (H0, W0)
        seg_logits_up = F.interpolate(seg_logits_s, size=(H0, W0), mode='bilinear', align_corners=False)

    # 还原训练状态
    model.train(model_was_training)
    return seg_logits_up