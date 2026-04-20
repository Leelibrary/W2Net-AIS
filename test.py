# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import numpy as np
import rasterio
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption, hyp_parse
from utils.utils import show_dota_results
from typing import Dict, Tuple, List
from utils.letterbox import letterbox_pair, unletterbox_map
from utils.lbr import estimate_lambda_geo_from_mask
import torch.nn.functional as F
from utils.geo_utils import pixel_segment_length_m
from pyproj import Transformer, Geod

# ===== 如果这些函数在你原 demo 文件里，建议直接 from demo import xxx，而不是重复写 =====
# from demo import (
#     overlay_seg_mask, generate_colors,
#     resolve_heading_by_largest_cc,
#     heading_from_quad, estimate_wavelength_from_mask, draw_heading_arrow
# )
# 为了方便你直接拷贝，这里先假定 overlay_seg_mask / generate_colors / resolve_heading_by_largest_cc
# 和你原文件一致（你可以直接复制过去或者改成 import）


DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }


# ========== 你原来 demo 里的几个函数（如已在别的文件，改成 import 即可） ==========
def apply_clahe_to_gray(gray, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    对单通道灰度图做 CLAHE，只用于模型输入，不改原图
    """
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    gray_eq = clahe.apply(gray)
    return gray_eq


def apply_clahe_to_bgr(bgr, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    对 BGR 彩色图按 L 通道做 CLAHE，只用于模型输入，不改原图
    """
    if bgr.dtype != np.uint8:
        bgr = cv2.normalize(bgr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    L_eq = clahe.apply(L)

    lab_eq = cv2.merge([L_eq, A, B])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return bgr_eq



def compass_wrap(deg):
    d = deg % 360.0
    if d < 0:
        d += 360.0
    if abs(d) < 1e-9 or abs(d - 360.0) < 1e-9:
        d = 0.0
    return d

def vec_to_compass_deg(vx, vy):
    # 图像坐标：x→右, y→下；罗盘角：北=0°、顺时针
    return compass_wrap(np.degrees(np.arctan2(vx, -vy)))

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def _long_edge_unit_from_quad(quad):
    """取 OBB 四点的“长边方向”（无向，图像坐标单位向量）"""
    q = quad.astype(np.float64)
    e0 = q[1] - q[0]
    e1 = q[2] - q[1]
    e2 = q[3] - q[2]
    e3 = q[0] - q[3]
    lens = [np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)]
    dirs = [e0, e1, e2, e3]
    return _unit(dirs[int(np.argmax(lens))])

def resolve_heading_by_largest_cc(bin_mask, obb_center, quad_pts,
                                  min_area=30, wake_is_behind=True, debug=False):
    """
    用“最大连通区域的质心”相对 OBB 中心，在长边轴的投影正负决定朝向。
    返回：heading_deg(float), v_img(np.array([vx,vy]))；若无有效CC返回(None, None)
    """
    H, W = bin_mask.shape[:2]
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    max_id, max_area_val = -1, -1
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and area > max_area_val:
            max_area_val = area
            max_id = i
    if max_id < 0:
        if debug:
            print("[largest-CC] no valid CC")
        return None, None

    cc_cx, cc_cy = float(cents[max_id, 0]), float(cents[max_id, 1])
    ship_cx, ship_cy = float(obb_center[0]), float(obb_center[1])

    u = _long_edge_unit_from_quad(quad_pts)  # 平行长边
    d = np.array([cc_cx - ship_cx, cc_cy - ship_cy], dtype=np.float64)
    s = float(d[0] * u[0] + d[1] * u[1])    # 投影

    if wake_is_behind:
        v = u if s < 0 else -u
    else:
        v = u if s >= 0 else -u

    heading_deg = vec_to_compass_deg(v[0], v[1])

    if debug:
        where = "behind(-)" if s < 0 else "ahead(+)"
        print(f"[largest-CC] area={max_area_val}  s={s:.1f} ({where})  heading={heading_deg:.2f}°")

    return float(heading_deg), np.array([float(v[0]), float(v[1])], dtype=np.float32)

def overlay_seg_mask(model, rgb_np, bgr_np, thr=0.6, alpha=0.4, img_size=640):
    H0, W0 = rgb_np.shape[:2]

    rgb_lb, _, ratio, pads = letterbox_pair(rgb_np, None, new_shape=img_size)

    im = rgb_lb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    im_t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float()
    if next(model.parameters()).is_cuda:
        im_t = im_t.cuda()

    with torch.no_grad():
        outs = model(im_t, return_seg=True)
        seg_logits = outs['seg_logits']              # (1,1,H_lb,W_lb)
        prob_lb = torch.sigmoid(seg_logits)[0, 0].float().cpu().numpy()

    prob = unletterbox_map(prob_lb, (H0, W0), ratio, pads)

    bin_mask = (prob > thr).astype(np.uint8)
    color_layer = np.zeros_like(bgr_np, dtype=np.uint8)
    color_layer[bin_mask == 1] = (0, 0, 255)
    vis = cv2.addWeighted(bgr_np, 1.0, color_layer, alpha, 0.0)

    # 简单统计
    num_cc, _, cc_stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    stats = {
        "pos_px": int(bin_mask.sum()),
        "area_ratio": float(bin_mask.sum()) / float(H0 * W0),
        "num_regions": int(max(0, num_cc - 1)),
        "max_area": int(cc_stats[1:, cv2.CC_STAT_AREA].max()) if num_cc > 1 else 0
    }
    return vis, bin_mask, stats

def generate_colors(dataset):
    num_colors = {
        'VOC': 20,
        'IC15': 1,
        'IC13': 1,
        'HRSC2016': 1,
        'DOTA': 15,
        'UCAS_AOD': 2,
        'NWPU_VHR': 10
    }
    if num_colors[dataset] == 1:
        colors = [(0, 255, 0)]
    elif num_colors[dataset] == 2:
        colors = [(0, 255, 0), (0, 0, 255)]
    else:
        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(num_colors[dataset])]
    return colors

# ========== 这三个如果你原来就有，可以删掉这里的定义，改成 import ==========

def heading_from_quad(quad):
    """
    根据 OBB 四点，取长边方向，转成罗盘角
    返回 heading_deg, v_img(单位向量 [vx,vy])
    """
    u = _long_edge_unit_from_quad(quad)
    heading_deg = vec_to_compass_deg(u[0], u[1])
    return float(heading_deg), np.array([float(u[0]), float(u[1])], dtype=np.float32)

def estimate_wavelength_from_mask(bin_mask, heading_rad, min_area=20):

    # 这里给个非常简单的示意实现（按投影方向做一维投影）
    H, W = bin_mask.shape
    # 投影方向单位向量（图像坐标）
    v = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float64)
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0:
        return {"lambda_px": None, "n_crests": 0}

    # 把连通区域像素投影到方向 v 上，做直方图观察周期
    proj = xs * v[0] + ys * v[1]
    proj = proj - proj.min()
    max_p = int(proj.max()) + 1
    hist, _ = np.histogram(proj, bins=max_p, range=(0, max_p))

    # 简单找峰值间距
    thresh = max(1, hist.max() * 0.3)
    peaks = np.where(hist > thresh)[0]
    if len(peaks) < 2:
        return {"lambda_px": None, "n_crests": len(peaks)}

    diffs = np.diff(peaks.astype(np.float32))
    lam = float(diffs.mean()) if len(diffs) > 0 else None
    return {"lambda_px": lam, "n_crests": int(len(peaks))}

def draw_heading_arrow(img, heading_deg, anchor=None,
                       length_ratio=0.15, color=(0, 0, 255), thickness=3):
    """
    在图上画箭头。heading_deg 为罗盘角（0=北，顺时针）
    anchor 为箭头起点像素坐标 (cx, cy)，默认图像中心
    """
    H, W = img.shape[:2]
    if anchor is None:
        cx, cy = W // 2, H // 2
    else:
        cx, cy = int(anchor[0]), int(anchor[1])

    theta = np.radians(heading_deg)
    # 罗盘角转图像方向向量
    vx = np.sin(theta)   # East
    vy = -np.cos(theta)  # North -> 图像 y 轴向下

    L = int(min(H, W) * length_ratio)
    x2 = int(cx + vx * L)
    y2 = int(cy + vy * L)

    cv2.arrowedLine(img, (cx, cy), (x2, y2), color, thickness, tipLength=0.2)


# ========== tiff 读取相关辅助函数 ==========

def is_tiff_file(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in ['.tif', '.tiff']

def load_tiff_for_model(path):
    """
    读取 tiff 并转换成:
        - src_bgr: 原图 BGR（用于可视化和保存）
        - im_rgb:  原图 RGB（如果你想，也可以不用它，用后面增强后的）
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取TIFF图像: {path}")

    # 单通道
    if img.ndim == 2:
        gray = img
        src_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        im_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 多通道
    elif img.ndim == 3:
        h, w, c = img.shape
        if c == 3:
            src_bgr = img
            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif c > 3:
            img3 = img[:, :, :3].copy()
            src_bgr = img3
            im_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        else:
            gray = img[:, :, 0]
            src_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            im_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        raise ValueError(f"不支持的 TIFF 维度: {img.shape}")

    return src_bgr, im_rgb



# ========== 核心：TIFF 推理 demo ==========

def demo_tiff(args):
    hyps = hyp_parse(args.hyp)
    ds = DATASETS[args.dataset](level=1)

    model = RetinaNet(backbone=args.backbone, hyps=hyps)
    if torch.cuda.is_available():
        model = model.cuda()

    colors = generate_colors(args.dataset)

    # 加载权重
    if args.weight.endswith('.pth'):
        chkpt = torch.load(args.weight, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        print('load weight from: {}'.format(args.weight))

    model.eval()

    out_dir = os.path.join('outputs', 'tiff')
    os.makedirs(out_dir, exist_ok=True)

    ims_list = [x for x in os.listdir(args.ims_dir) if is_tiff_file(x)]
    print(f"Found {len(ims_list)} tiff images in {args.ims_dir}")

    t0 = time.time()

    with torch.no_grad():
        for idx, im_name in enumerate(ims_list):
            t = time.time()
            im_path = os.path.join(args.ims_dir, im_name)
            s = 'image %g/%g %s: ' % (idx, len(ims_list), im_path)

            try:
                src, im_rgb_raw = load_tiff_for_model(im_path)  # src = 原图BGR
            except Exception as e:
                print(f"[ERROR] {im_path}: {e}")
                continue

            bgr0 = src.copy()  # 原图备份，可视化用

            # ===== 只给模型看的增强版 =====
            # 用 CLAHE 增强一个 BGR 版本，供检测用
            bgr_for_model = apply_clahe_to_bgr(src)
            im_rgb_for_model = cv2.cvtColor(bgr_for_model, cv2.COLOR_BGR2RGB)

            # 分割输入：可以选择灰度 + CLAHE，或者直接用增强后的彩色
            if getattr(args, "demo_gray", False):
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 原图转灰度
                gray_eq = apply_clahe_to_gray(gray)  # 只给模型看的增强
                seg_rgb = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2RGB)
            else:
                # 用和检测一样的增强图做分割输入
                seg_rgb = im_rgb_for_model

            # 1) 分割 + 连通域
            src, bin_mask, seg_stats = overlay_seg_mask(
                model, seg_rgb, src,  # seg_rgb = CLAHE 后的输入；src = 原图
                thr=0.6,
                alpha=0.4,
                img_size=args.target_size[0]
            )

            # 2) 检测（OBB）
            cls_dets = im_detect(model, im_rgb_for_model, target_sizes=args.target_size)

            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                bbox = cls_dets[j, 2:]
                if len(bbox) == 4:
                    draw_caption(src, bbox, '{:1.3f}'.format(scores))
                    cv2.rectangle(
                        src,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color=(0, 0, 255),
                        thickness=1
                    )
                else:
                    pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, thickness=1, color=colors[int(cls - 1)])

                    put_label = True
                    plot_anchor = False
                    if put_label:
                        label = ds.return_class(cls) + str(' %.2f' % scores)
                        fontScale = 0.45
                        font = cv2.FONT_HERSHEY_COMPLEX
                        thickness = 1
                        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                        c1 = tuple(bbox[:2].astype('int'))
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5

                        cv2.rectangle(src, c1, c2, colors[int(cls - 1)], -1)
                        cv2.putText(
                            src, label, (c1[0], c1[1] - 4),
                            font, fontScale, [0, 0, 0],
                            thickness=thickness, lineType=cv2.LINE_AA
                        )

                        if plot_anchor and len(bbox) >= 9:
                            pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                            cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)

            # 3) 航向 + 波长（和你原 demo 完全同一套）
            heading_rad_raw = None
            best_area = -1.0
            best_quad = None
            best_center = None

            for j in range(len(cls_dets)):
                bbox = cls_dets[j, 2:]
                if len(bbox) >= 5:
                    cx, cy, w, h, ang = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
                    area = w * h
                    if area > best_area:
                        best_area = area
                        heading_rad_raw = ang
                        quad = rbox_2_quad(bbox[:5]).reshape((4, 2))
                        best_quad = quad
                        best_center = (cx, cy)

            if best_quad is not None:
                heading_deg, v_img = heading_from_quad(best_quad)
                heading_deg_quad, v_img_quad = heading_deg, v_img

                print(
                    f"    [DEBUG] raw heading_rad(model) = {heading_rad_raw:.6f} rad  ({heading_rad_raw / np.pi:.3f} π)"
                    if heading_rad_raw is not None else
                    "    [DEBUG] raw heading_rad(model) = None"
                )
                print(f"    [DEBUG] heading_deg(from quad) = {heading_deg:.2f}°")

                # 这里用“最大连通域”判向，你也可以换成你原来的 resolve_heading_by_first_last
                cc_heading_deg, cc_v_img = resolve_heading_by_largest_cc(
                    bin_mask=bin_mask,
                    obb_center=best_center,
                    quad_pts=best_quad,
                    min_area=100,
                    wake_is_behind=True,
                    debug=True
                )

                if cc_heading_deg is not None:
                    heading_deg, v_img = cc_heading_deg, cc_v_img
                else:
                    heading_deg, v_img = heading_deg_quad, v_img_quad

                print(f"    [CC-longedge] display heading = {heading_deg:.2f}°")

                # ====== 逐个波峰 + 经纬度距离 的 λ 估计 ======
                wl_info = estimate_lambda_geo_from_mask(
                    bin_mask=bin_mask,
                    v_img=v_img,  # 这里用最终的航向向量（和尾迹方向相关）
                    center=best_center,
                    tif_path=im_path,  # 当前这张 TIFF 的路径
                    min_run_px=3.0,
                    min_pairs=2
                )
                lam_px = wl_info["lambda_px"]
                lam_m = wl_info["lambda_m"]
                n_crests = wl_info["n_crests"]
                # ===========================================

                # ====== 像素波长 → 实际距离（米） ====== 1126
                lam_m = None
                if lam_px is not None and best_center is not None:

                    col1 = float(best_center[0])
                    row1 = float(best_center[1])

                    # 沿波向走 lam_px 像素
                    col2 = col1 + float(v_img[0]) * float(lam_px)
                    row2 = row1 + float(v_img[1]) * float(lam_px)

                    try:
                        lam_m = pixel_segment_length_m(
                            im_path,  # 当前 tiff 的绝对路径
                            col1, row1,
                            col2, row2
                        )
                    except Exception as e:
                        print(f"[WARN] λ 像素转距离失败: {e}")
                        lam_m = None

                if getattr(args, "show_waveinfo", False):
                    txt = [f"heading={heading_deg:.1f}°"]

                    # txt.append(f"λ={lam_px:.1f}px" if lam_px is not None else "λ=NA")
                    # ===== λ 显示（px + m）=====
                    if lam_px is not None:
                        if lam_m is not None:
                            txt.append(f"λ={lam_px:.1f}px / {lam_m:.2f}m")
                        else:
                            txt.append(f"λ={lam_px:.1f}px")
                    else:
                        txt.append("λ=NA")
                    # ==========================

                    txt.append(f"crests={n_crests}")
                    cv2.putText(
                        src, " / ".join(txt), (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA
                    )

                    if best_center is not None:
                        draw_heading_arrow(
                            src, heading_deg,
                            anchor=best_center,
                            length_ratio=0.15,
                            color=(0, 0, 255),
                            thickness=3
                        )
                    else:
                        draw_heading_arrow(
                            src, heading_deg,
                            length_ratio=0.12,
                            color=(0, 0, 255),
                            thickness=3
                        )

                    # print(
                    #     f"    Heading={heading_deg:.1f}°  "
                    #     f"| λ={lam_px if lam_px is not None else 'NA'} px "
                    #     f"| crests={n_crests}"
                    # )

                    if lam_m is not None:
                        print(
                            f"    Heading={heading_deg:.1f}°  "
                            f"| λ={lam_px if lam_px is not None else 'NA'} px "
                            f"| λ_m={lam_m:.2f} m "
                            f"| crests={n_crests}"
                        )
                    else:
                        print(
                            f"    Heading={heading_deg:.1f}°  "
                            f"| λ={lam_px if lam_px is not None else 'NA'} px "
                            f"| crests={n_crests}"
                        )

            # 4) 日志与保存
            print('%sDone. (%.3fs) %d objs | seg: area=%d px (%.1f%%), blobs=%d, max_blob=%d px'
                  % (s, time.time() - t, len(cls_dets),
                     seg_stats["pos_px"],
                     100.0 * seg_stats["area_ratio"],
                     seg_stats["num_regions"],
                     seg_stats["max_area"]))

            base_name = os.path.splitext(os.path.basename(im_path))[0] + ".png"
            out_path = os.path.join(out_dir, base_name)
            cv2.imwrite(out_path, src)

    print('All TIFF done. (%.3fs)' % (time.time() - t0))


def parse_args():
    parser = argparse.ArgumentParser(description='TIFF inference demo')

    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--dataset', type=str, default='HRSC2016')
    parser.add_argument('--backbone', type=str, default='fca101')
    parser.add_argument('--weight', type=str, default='weights/best_1126.pth')
    parser.add_argument('--ims_dir', type=str, default='/home/lab/libr/obb-RetinaNet/wave_dataset/test/tiff', help='tiff images directory')
    parser.add_argument('--target_size', nargs='+', type=int, default=[1280])
    parser.add_argument('--demo_gray', action='store_true',
                        help='use gray->RGB for segmentation branch')
    parser.add_argument('--show_waveinfo', default=True, help='是否在结果图和控制台输出中显示航向角与横波波长信息')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    demo_tiff(args)
