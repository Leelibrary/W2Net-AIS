import re
import torch
import cv2
import numpy as np


class SimpleGradCAM:
    """
    通用 Grad-CAM（不依赖第三方库）
    - forward hook 抓 activation
    - 在 activation 上注册梯度 hook 抓 gradient
    - cam = ReLU( sum_c( mean_hw(grad_c) * act_c ) )
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handle = None

    def _forward_hook(self, module, inp, out):
        self.activations = out

        def _grad_hook(grad):
            self.gradients = grad

        out.register_hook(_grad_hook)

    def __enter__(self):
        self.hook_handle = self.target_layer.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def _pick_target_scalar(self, outputs):
        """
        RetinaNet/OBB 检测模型输出形式不固定：
        tensor / list/tuple / dict 都尝试兼容。
        默认取“最大值”做反传（先保证能跑通）。
        """
        if torch.is_tensor(outputs):
            return outputs.max()

        if isinstance(outputs, (list, tuple)):
            for x in outputs:
                if torch.is_tensor(x):
                    return x.max()
            raise RuntimeError("outputs(list/tuple)里没有 tensor")

        if isinstance(outputs, dict):
            for k in ["cls", "logits", "scores", "cls_logits", "conf"]:
                if k in outputs and torch.is_tensor(outputs[k]):
                    return outputs[k].max()
            for v in outputs.values():
                if torch.is_tensor(v):
                    return v.max()
            raise RuntimeError("outputs(dict)里没有 tensor")

        raise RuntimeError(f"不支持的 outputs 类型: {type(outputs)}")

    def compute_cam(self, input_tensor, target_scalar=None):
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(input_tensor)
        if target_scalar is None:
            target_scalar = self._pick_target_scalar(outputs)

        target_scalar.backward(retain_graph=False)

        act = self.activations  # (B,C,h,w)
        grad = self.gradients   # (B,C,h,w)
        if act is None or grad is None:
            raise RuntimeError("没有拿到 activation/gradient：target_layer 选得不对 or 该层没参与前向")

        w = grad.mean(dim=(2, 3), keepdim=True)      # (B,C,1,1)
        cam = (w * act).sum(dim=1)                   # (B,h,w)
        cam = torch.relu(cam)

        cam = cam[0]
        cam -= cam.min()
        cam /= (cam.max() + 1e-6)
        return cam.detach().cpu().numpy()


def overlay_cam_on_bgr(img_bgr, cam01, alpha=0.45):
    """
    img_bgr: uint8 (H,W,3)
    cam01: float (h,w) in [0,1]
    """
    H, W = img_bgr.shape[:2]
    heat = (cam01 * 255).astype(np.uint8)
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)
    return overlay, heat_color


def find_cam_layer_for_fca(model, prefer_keywords=("layer4", "stage4", "c5", "out5", "p5", "fpn")):
    """
    自动帮你在 fca101(backbone) 里找一个合适的 CAM 层：
    规则：
      1) 优先找名字里包含 prefer_keywords 的 Conv2d/Sequential
      2) 找不到就退化为：backbone 内最后一个 Conv2d
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise RuntimeError("model 没有 backbone 属性，无法自动选层")

    named = list(backbone.named_modules())

    # 1) 优先匹配关键词（从后往前找，越靠后语义越强）
    for name, m in reversed(named):
        if any(k.lower() in name.lower() for k in prefer_keywords):
            if isinstance(m, torch.nn.Conv2d):
                return m, f"backbone.{name}"
            if isinstance(m, torch.nn.Sequential):
                # sequential 里找最后一个 conv
                for subname, sm in reversed(list(m.named_modules())):
                    if isinstance(sm, torch.nn.Conv2d):
                        return sm, f"backbone.{name}.{subname}"

    # 2) 退化：找最后一个 Conv2d
    for name, m in reversed(named):
        if isinstance(m, torch.nn.Conv2d):
            return m, f"backbone.{name}"

    raise RuntimeError("backbone 里找不到 Conv2d，无法做 Grad-CAM")


def dump_backbone_conv_candidates(model, topk=30):
    """
    打印 backbone 里靠后的 conv 候选，方便你人工指定。
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        print("[CAM] model 没有 backbone")
        return
    convs = [(n, m) for n, m in backbone.named_modules() if isinstance(m, torch.nn.Conv2d)]
    print(f"\n[CAM] backbone Conv2d 总数: {len(convs)} | 显示最后 {min(topk, len(convs))} 个：")
    for n, m in convs[-topk:]:
        print(f"  - backbone.{n}: {tuple(m.weight.shape)}")
    print()
