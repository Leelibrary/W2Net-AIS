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

from pathlib import Path
import re

def safe_name(s: str):
    s = re.sub(r"[^\w\-_\.]+", "_", s)
    return s[:200]

def normalize_percentile(heat: np.ndarray, lo=1, hi=99):
    a = np.percentile(heat, lo)
    b = np.percentile(heat, hi)
    heat = np.clip(heat, a, b)
    heat = (heat - a) / (b - a + 1e-6)
    return heat

def tensor_to_heat_u8(feat_bchw: torch.Tensor, reduce="mean"):
    """
    feat_bchw: (1,C,H,W)
    return: (H,W) uint8 heatmap
    """
    feat = feat_bchw[0].detach()

    # 1) 通道压缩
    if reduce == "mean":
        heat = feat.mean(dim=0)
    elif reduce == "max":
        heat = feat.max(dim=0)[0]
    elif reduce == "l2":
        heat = torch.sqrt((feat ** 2).sum(dim=0) + 1e-6)
    else:
        raise ValueError("reduce must be mean/max/l2")

    # 2) 压制弱响应背景（⭐关键就在这里⭐）
    # heat = heat - heat.median()     # 或 heat.mean()
    heat = heat - heat.quantile(0.7)
    heat = torch.relu(heat)

    # 3) 归一化（百分位裁剪，强烈推荐）
    heat_np = heat.cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(heat_np, 2), np.percentile(heat_np, 98)
    heat_np = np.clip(heat_np, lo, hi)
    heat_np = (heat_np - lo) / (hi - lo + 1e-6)

    heat_u8 = (heat_np * 255.0).clip(0, 255).astype(np.uint8)
    return heat_u8


def overlay_heat_on_bgr(img_bgr: np.ndarray, heat_u8: np.ndarray, alpha=0.45):
    H, W = img_bgr.shape[:2]
    heat_rs = cv2.resize(heat_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(heat_rs, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)
    return overlay, heat_color

def collect_conv_layers(model, max_layers=0, keyword=""):
    """
    收集需要可视化的层：默认所有 Conv2d
    max_layers=0 表示不限制；>0 表示只取最后 max_layers 个（强烈推荐）
    keyword 非空则只保留名字包含 keyword 的层
    return: list[(name,module)]
    """
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if keyword and (keyword.lower() not in name.lower()):
                continue
            layers.append((name, m))

    if max_layers and max_layers > 0:
        layers = layers[-max_layers:]
    return layers

def run_and_dump_feature_heatmaps(
    model,
    input_tensor,         # (1,3,H,W) float
    img_bgr_for_overlay,  # 原图 BGR (H,W,3) uint8，用于叠加
    save_dir,
    best_quad,
    layers,               # list[(name,module)]
    reduce="mean",
    alpha=0.45,
    save_raw_heat=False
):
    """
    对指定 layers 做 forward hook，抓每层输出并保存热力图。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    quad = best_quad
    feats = {}
    hooks = []

    def make_hook(layer_name):
        def _hook(m, inp, out):
            # out 可能不是 Tensor（少见），过滤一下
            if torch.is_tensor(out) and out.ndim == 4:
                # 立刻转到 CPU，降低显存占用
                feats[layer_name] = out.detach().cpu()
        return _hook

    # 注册 hook
    for name, m in layers:
        hooks.append(m.register_forward_hook(make_hook(name)))

    # 前向（不需要梯度）
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    print("=== DEBUG model output ===")
    print(type(_))
    if torch.is_tensor(_):
        print(_.shape)
    elif isinstance(_, (list, tuple)):
        for i, x in enumerate(_):
            if torch.is_tensor(x):
                print(i, x.shape)
    elif isinstance(_, dict):
        for k, v in _.items():
            if torch.is_tensor(v):
                print(k, v.shape)
    print("==========================")
    # 解除 hook
    for h in hooks:
        h.remove()

    # 保存
    names = list(feats.keys())
    names.sort(key=lambda x: x)  # 名字排序；你也可以用 layers 的顺序
    # 按 layers 顺序保存更直观：
    ordered = [n for n, _ in layers if n in feats]

    for i, layer_name in enumerate(ordered):
        feat = feats[layer_name]  # (1,C,h,w)
        heat_feat = tensor_to_heat_u8(feat)  # (h_feat, w_feat), uint8

        # 2) resize 到原图大小
        H, W = img_bgr_for_overlay.shape[:2]
        heat_img = cv2.resize(
            heat_feat,
            (W, H),
            interpolation=cv2.INTER_LINEAR
        )  # (H, W)

        # 5) overlay
        overlay, _ = overlay_heat_on_bgr(
            img_bgr_for_overlay,
            heat_img,
            alpha=alpha
        )

        base = f"{i:03d}_{safe_name(layer_name)}"
        cv2.imwrite(str(save_dir / f"{base}.png"), overlay)


def make_soft_vicinity_weight(img_shape, quad, max_dist=80):
    """
    返回一个 [0,1] 的 soft weight
    框内 ≈1，向外线性/高斯衰减
    """
    H, W = img_shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(quad, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # 距离变换
    dist = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5)
    w = np.clip(1.0 - dist / max_dist, 0.0, 1.0)

    return w.astype(np.float32)

