import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from utils.letterbox import letterbox_pair, unletterbox_map
from utils.lbr import *
import os
import random
import re




def get_obb_direction_vector(quad_pts):
    """从OBB四点获取长边方向向量"""
    q = quad_pts.astype(np.float64)
    edges = [q[1] - q[0], q[2] - q[1], q[3] - q[2], q[0] - q[3]]
    lens = [np.linalg.norm(e) for e in edges]
    long_edge = edges[int(np.argmax(lens))]
    return long_edge / (np.linalg.norm(long_edge) + 1e-9)


def find_wave_edge_point(mask_region, v_direction, center_pt, mode='leading'):
    """找到连通域沿指定方向的边缘点"""
    ys, xs = np.where(mask_region > 0)
    if len(xs) == 0:
        return None

    points = np.column_stack([xs, ys])
    ref = np.array(center_pt, dtype=np.float64)
    projections = (points - ref) @ v_direction

    idx = np.argmax(projections) if mode == 'leading' else np.argmin(projections)
    return (float(xs[idx]), float(ys[idx]))


def extract_wave_crests_with_edges(bin_mask, v_obb, obb_center, min_area=50, debug=False):
    """提取所有横波连通域及边缘点"""
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8
    )

    wave_crests = []

    for i in range(1, num_cc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
        mask_i = (labels == i).astype(np.uint8)

        leading_pt = find_wave_edge_point(mask_i, v_obb, obb_center, mode='leading')
        trailing_pt = find_wave_edge_point(mask_i, v_obb, obb_center, mode='trailing')

        if leading_pt is None or trailing_pt is None:
            continue

        vec = np.array([cx - obb_center[0], cy - obb_center[1]], dtype=np.float64)
        projection = float(vec @ v_obb)

        wave_crests.append({
            'id': i,
            'area': area,
            'center': (cx, cy),
            'leading_edge': leading_pt,
            'trailing_edge': trailing_pt,
            'projection': projection
        })

    wave_crests.sort(key=lambda x: x['projection'])

    if debug:
        print(f"[提取波峰] 找到 {len(wave_crests)} 个有效波峰 (min_area={min_area}px)")

    return wave_crests

def calculate_wavelength_from_edges(bin_mask, v_obb, obb_center, tif_path,
                                    min_area=50, edge_mode='leading_to_leading',
                                    debug=False):
    """
    基于OBB方向计算波长

    Args:
        edge_mode: 'leading_to_leading' | 'trailing_to_trailing' | 'trailing_to_leading'
    """
    wave_crests = extract_wave_crests_with_edges(
        bin_mask, v_obb, obb_center, min_area, debug
    )

    if len(wave_crests) < 2:
        return {
            'wavelength_m': None,
            'wavelengths_m': [],
            'std_m': None,
            'n_crests': len(wave_crests),
            'wave_crests': wave_crests
        }

    wavelengths = []

    try:
        for i in range(len(wave_crests) - 1):
            curr_wave = wave_crests[i]
            next_wave = wave_crests[i + 1]

            if edge_mode == 'leading_to_leading':
                pt1, pt2 = curr_wave['leading_edge'], next_wave['leading_edge']
            elif edge_mode == 'trailing_to_trailing':
                pt1, pt2 = curr_wave['trailing_edge'], next_wave['trailing_edge']
            else:  # trailing_to_leading
                pt1, pt2 = curr_wave['trailing_edge'], next_wave['leading_edge']

            lon1, lat1 = pixel_to_lonlat(tif_path, pt1[0], pt1[1])
            lon2, lat2 = pixel_to_lonlat(tif_path, pt2[0], pt2[1])

            dist_m = calculate_geo_distance(lon1, lat1, lon2, lat2)
            wavelengths.append(dist_m)

            if debug:
                print(f"    波{i + 1}→波{i + 2}: {dist_m:.2f}m")

    except Exception as e:
        print(f"[错误] 计算波长失败: {e}")
        return {
            'wavelength_m': None,
            'wavelengths_m': [],
            'std_m': None,
            'n_crests': len(wave_crests),
            'wave_crests': wave_crests
        }

    if len(wavelengths) > 0:
        wavelength_m = float(np.mean(wavelengths))
        std_m = float(np.std(wavelengths))
    else:
        wavelength_m = None
        std_m = None

    return {
        'wavelength_m': wavelength_m,
        'wavelengths_m': wavelengths,
        'std_m': std_m,
        'n_crests': len(wave_crests),
        'wave_crests': wave_crests
    }



# ============ 辅助函数 ============

def apply_clahe_to_gray(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(gray)


def apply_clahe_to_bgr(bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    if bgr.dtype != np.uint8:
        bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    L_eq = clahe.apply(L)
    lab_eq = cv2.merge([L_eq, A, B])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def compass_wrap(deg):
    d = deg % 360.0
    if d < 0:
        d += 360.0
    return d


def vec_to_compass_deg(vx, vy):
    return compass_wrap(np.degrees(np.arctan2(vx, -vy)))


def heading_from_obb_vector(v_obb):
    """从OBB向量计算罗盘航向"""
    heading_deg = vec_to_compass_deg(v_obb[0], v_obb[1])
    return float(heading_deg)


def draw_heading_arrow(img, heading_deg, anchor=None, length_ratio=0.15,
                       color=(0, 0, 255), thickness=3):
    H, W = img.shape[:2]
    if anchor is None:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = int(anchor[0]), int(anchor[1])

    theta = np.radians(heading_deg)
    vx = np.sin(theta)
    vy = -np.cos(theta)

    L = int(min(H, W) * length_ratio)
    x2 = int(cx + vx * L)
    y2 = int(cy + vy * L)

    cv2.arrowedLine(img, (cx, cy), (x2, y2), color, thickness, tipLength=0.2)


def overlay_seg_mask(model, rgb_np, bgr_np, thr=0.6, alpha=0.4, img_size=640, color=(0, 0, 255)):
    H0, W0 = rgb_np.shape[:2]
    rgb_lb, _, ratio, pads = letterbox_pair(rgb_np, None, new_shape=img_size)

    im = rgb_lb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    im_t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float()

    if next(model.parameters()).is_cuda:
        im_t = im_t.cuda()

    with torch.no_grad():
        outs = model(im_t, return_seg=True)
        seg_logits = outs['seg_logits']
        prob_lb = torch.sigmoid(seg_logits)[0, 0].float().cpu().numpy()

    prob = unletterbox_map(prob_lb, (H0, W0), ratio, pads)
    bin_mask = (prob > thr).astype(np.uint8)

    color_layer = np.zeros_like(bgr_np, dtype=np.uint8)
    color_layer[bin_mask == 1] = color
    vis = cv2.addWeighted(bgr_np, 1.0, color_layer, alpha, 0.0)

    num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    stats = {
        "pos_px": int(bin_mask.sum()),
        "area_ratio": float(bin_mask.sum()) / float(H0 * W0),
        "num_regions": int(max(0, num_cc - 1)),
        "max_area": int(cc_stats[1:, cv2.CC_STAT_AREA].max()) if num_cc > 1 else 0
    }
    return vis, bin_mask, stats


def generate_colors(dataset):
    num_colors = {'VOC': 20, 'IC15': 1, 'IC13': 1, 'HRSC2016': 1,
                  'DOTA': 15, 'UCAS_AOD': 2, 'NWPU_VHR': 10}
    if num_colors[dataset] == 1:
        return [(0, 255, 0)]
    elif num_colors[dataset] == 2:
        return [(0, 255, 0), (0, 0, 255)]
    else:
        return [[random.randint(0, 255) for _ in range(3)]
                for _ in range(num_colors[dataset])]


def is_tiff_file(name):
    return os.path.splitext(name)[1].lower() in ['.tif', '.tiff']


def load_tiff_for_model(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取TIFF: {path}")

    if img.ndim == 2:
        src_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3:
        if img.shape[2] >= 3:
            img3 = img[:, :, :3].copy()
            src_bgr = img3
            im_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        else:
            gray = img[:, :, 0]
            src_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            im_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"不支持的TIFF维度: {img.shape}")

    return src_bgr, im_rgb


def correct_heading_by_cc_extreme(
    bin_mask: np.ndarray,
    quad: np.ndarray,
    v_obb: np.ndarray,
    heading_deg: float,
    min_cc_area: int = 30,
    choose_small: str = "smallest",   # "smallest" or "farthest"
    debug: bool = False
):
    """
    用 OBB 内部的分割连通域结构来消除 180° 航向模糊。

    核心：
      1) 在 OBB polygon 内取 bin_mask
      2) 找连通域，过滤太小的噪声
      3) 取最大连通域(面积最大)的质心 Cmax
      4) 取最小连通域(面积最小)的质心 Cmin（或按 farthest 选与 Cmax 最远者）
      5) v_cc = Cmax - Cmin （你说的“最大到最小”方向）
      6) 若 dot(v_obb, v_cc) < 0，则 heading + 180

    返回：
      heading_corr_deg, flipped(bool), info(dict)
    """
    H, W = bin_mask.shape[:2]

    # --- 1) 构造 OBB polygon mask ---
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 1)

    # --- 2) 只保留 OBB 内部的分割 ---
    inner = (bin_mask.astype(np.uint8) > 0).astype(np.uint8)
    inner = (inner & poly_mask).astype(np.uint8)

    # 连通域（8邻域）
    num, labels, stats, cents = cv2.connectedComponentsWithStats(inner, connectivity=8)

    # stats: [label, x, y, w, h, area] 其中 stats[i, cv2.CC_STAT_AREA] 是面积
    areas = stats[:, cv2.CC_STAT_AREA]  # 包含背景 label=0

    # --- 3) 过滤：去背景 + 去小噪声 ---
    valid = []
    for lab in range(1, num):
        a = int(areas[lab])
        if a >= int(min_cc_area):
            cx, cy = cents[lab]
            valid.append((lab, a, float(cx), float(cy)))

    if len(valid) < 2:
        # 连通域不足，无法做“最大-最小”方向
        if debug:
            print(f"[cc-correct] valid components < 2 (valid={len(valid)}) -> skip")
        return float(heading_deg), False, {"reason": "not_enough_components", "valid": len(valid)}

    # --- 4) 最大连通域 ---
    valid.sort(key=lambda x: x[1], reverse=True)
    lab_max, amax, cx_max, cy_max = valid[0]

    # --- 5) 选择“最小”连通域（或最远）---
    if choose_small == "smallest":
        lab_min, amin, cx_min, cy_min = min(valid[1:], key=lambda x: x[1])
    else:
        # 与最大质心最远的那个（有时比“最小”更稳，避免噪声小块）
        lab_min, amin, cx_min, cy_min = max(
            valid[1:],
            key=lambda x: (x[2] - cx_max) ** 2 + (x[3] - cy_max) ** 2
        )

    # --- 6) v_cc：你定义的“最大到最小”方向 ---
    v_cc = np.array([cx_max - cx_min, cy_max - cy_min], dtype=np.float64)
    v_cc = v_cc / (np.linalg.norm(v_cc) + 1e-9)

    # --- 7) 与 v_obb 比较方向，决定翻转 ---
    v_obb_n = v_obb.astype(np.float64)
    v_obb_n = v_obb_n / (np.linalg.norm(v_obb_n) + 1e-9)

    dot = float(v_obb_n[0] * v_cc[0] + v_obb_n[1] * v_cc[1])
    flipped = (dot < 0)

    heading_corr = (heading_deg + 180.0) % 360.0 if flipped else float(heading_deg)

    if debug:
        print(f"[cc-correct] amax={amax}, amin={amin}, dot={dot:.4f}, flipped={flipped}, heading={heading_deg:.1f}->{heading_corr:.1f}")

    info = {
        "amax": amax, "amin": amin,
        "cmax": (cx_max, cy_max),
        "cmin": (cx_min, cy_min),
        "dot": dot,
        "flipped": flipped,
        "choose_small": choose_small
    }
    return float(heading_corr), bool(flipped), info

def correct_heading_by_cc_topk_vote(
    bin_mask: np.ndarray,
    quad: np.ndarray,
    v_obb: np.ndarray,
    heading_deg: float,
    top_k: int = 8,
    min_cc_area: int = 30,
    dot_eps: float = 0.05,   # |dot| 太小认为“不确定”，不参与投票
    debug: bool = False
):
    """
    用 OBB 内部 Top-K 连通域做“方向投票”，以最大连通域为基准消除 180° 航向模糊。

    投票规则：
      - 取最大连通域质心 C0
      - 对 i=2..K 计算 v_i = Ci - C0
      - sign_i = sign(dot(v_i, v_obb))
      - dot>0 归为 A，dot<0 归为 B（两类相差180°）
      - 多数票决定是否翻转 heading 180°

    返回：
      heading_corr_deg, flipped(bool), info(dict)
    """
    H, W = bin_mask.shape[:2]

    # --- 1) OBB polygon mask ---
    poly = quad.astype(np.int32).reshape(-1, 1, 2)
    poly_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [poly], 1)

    # --- 2) OBB 内部 mask ---
    inner = (bin_mask.astype(np.uint8) > 0).astype(np.uint8)
    inner = (inner & poly_mask).astype(np.uint8)

    num, labels, stats, cents = cv2.connectedComponentsWithStats(inner, connectivity=8)

    # 有效连通域列表（去背景 + 过滤小噪声）
    comps = []
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_cc_area):
            cx, cy = cents[lab]
            comps.append((lab, area, float(cx), float(cy)))

    if len(comps) < 2:
        if debug:
            print(f"[cc-vote] valid comps < 2 (valid={len(comps)}) -> skip")
        return float(heading_deg), False, {"reason": "valid_cc<2", "valid": len(comps)}

    # --- 3) 取 Top-K ---
    comps.sort(key=lambda x: x[1], reverse=True)
    comps = comps[:min(top_k, len(comps))]

    # 最大连通域为基准
    lab0, a0, cx0, cy0 = comps[0]

    # 归一化 v_obb
    v_obb_n = np.array(v_obb, dtype=np.float64)
    v_obb_n = v_obb_n / (np.linalg.norm(v_obb_n) + 1e-9)

    # --- 4) 投票 ---
    votes_pos = 0  # A 类：dot >= +dot_eps
    votes_neg = 0  # B 类：dot <= -dot_eps
    skipped   = 0  # dot 太小不投票

    details = []
    for (lab, area, cxi, cyi) in comps[1:]:
        v_i = np.array([cxi - cx0, cyi - cy0], dtype=np.float64)
        nrm = np.linalg.norm(v_i)
        if nrm < 1e-6:
            skipped += 1
            details.append((lab, area, 0.0, "skip_zero"))
            continue
        v_i /= (nrm + 1e-9)

        dot = float(v_i[0] * v_obb_n[0] + v_i[1] * v_obb_n[1])

        if dot >= dot_eps:
            votes_pos += 1
            details.append((lab, area, dot, "A(pos)"))
        elif dot <= -dot_eps:
            votes_neg += 1
            details.append((lab, area, dot, "B(neg)"))
        else:
            skipped += 1
            details.append((lab, area, dot, "skip_small"))

    # 如果有效票太少，跳过
    valid_votes = votes_pos + votes_neg
    if valid_votes == 0:
        if debug:
            print("[cc-vote] no valid votes -> skip")
        return float(heading_deg), False, {
            "reason": "no_valid_votes",
            "votes_pos": votes_pos, "votes_neg": votes_neg, "skipped": skipped,
            "top_k": len(comps)
        }

    # --- 5) 多数决策：负票多 -> 翻转 180° ---
    # 平票：不翻（更保守）
    flipped = (votes_neg > votes_pos)
    heading_corr = (heading_deg + 180.0) % 360.0 if flipped else float(heading_deg)

    if debug:
        print(f"[cc-vote] top_k={len(comps)} a0={a0} | pos={votes_pos} neg={votes_neg} skip={skipped} "
              f"-> flipped={flipped} | {heading_deg:.1f}->{heading_corr:.1f}")
        for lab, area, dot, tag in details:
            print(f"    - lab={lab} area={area} dot={dot:+.3f} => {tag}")

    info = {
        "top_k": len(comps),
        "base": {"lab": lab0, "area": a0, "c0": (cx0, cy0)},
        "votes_pos": votes_pos,
        "votes_neg": votes_neg,
        "skipped": skipped,
        "dot_eps": dot_eps,
        "details": details,
        "flipped": flipped
    }
    return float(heading_corr), bool(flipped), info
