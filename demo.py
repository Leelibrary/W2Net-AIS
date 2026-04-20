from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import numpy as np
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import *
from utils.utils import is_image, draw_caption, hyp_parse
from utils.utils import show_dota_results
from eval import evaluate
from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge
import torch.nn.functional as F
from utils.letterbox import letterbox_pair, unletterbox_map


DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }
def overlay_ccs_on_image(base_bgr, bin_mask, alpha=0.45, min_area=20, draw_ids=True):
    """
    将 bin_mask 的每个连通区域以不同颜色画到 base_bgr 上。
    - base_bgr: 原图(BGR)
    - bin_mask: 二值掩膜 (H,W) 0/1
    - alpha: 颜色层与原图的融合比例
    - min_area: 过滤过小噪声
    - draw_ids: 是否在质心处标注编号

    返回:
      vis: 叠加后的可视化图像(BGR)
    """
    H, W = bin_mask.shape[:2]
    # 获取连通域
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8
    )

    # 生成颜色：均匀分布在HSV色环，再转BGR，保证清晰区分
    # index从1开始(0是背景)
    colors = []
    n_comp = max(0, num - 1)
    for i in range(n_comp):
        hue = int(180.0 * i / max(1, n_comp))  # OpenCV HSV: H∈[0,180)
        col_hsv = np.uint8([[[hue, 200, 255]]])  # S高一点, V高一点
        col_bgr = cv2.cvtColor(col_hsv, cv2.COLOR_HSV2BGR)[0,0].tolist()
        colors.append((int(col_bgr[0]), int(col_bgr[1]), int(col_bgr[2])))

    color_layer = np.zeros_like(base_bgr, dtype=np.uint8)

    valid_id = 0
    for i in range(1, num):  # 1..num-1
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        # 该连通域的颜色
        if n_comp > 0:
            col = colors[(i - 1) % n_comp]
        else:
            col = (0, 0, 255)

        # 填色：把 labels==i 的像素设为该颜色
        mask_i = (labels == i)
        color_layer[mask_i] = col

        # 可选：画边界更清晰（先提取轮廓）
        # 注意：findContours 需要 uint8 0/255
        cnt_mask = np.zeros((H, W), dtype=np.uint8)
        cnt_mask[mask_i] = 255
        contours, _ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(color_layer, contours, -1, (0,0,0), thickness=1)

        # 可选：在质心处标注编号
        if draw_ids:
            cx, cy = float(cents[i,0]), float(cents[i,1])
            cv2.putText(color_layer, str(valid_id+1), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(color_layer, str(valid_id+1), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            valid_id += 1

    # 与原图融合
    vis = cv2.addWeighted(base_bgr, 1.0, color_layer, alpha, 0.0)
    return vis

def compass_wrap(deg):
    d = deg % 360.0
    if d < 0: d += 360.0
    if abs(d) < 1e-9 or abs(d-360.0) < 1e-9: d = 0.0
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
    e0 = q[1] - q[0]; e1 = q[2] - q[1]; e2 = q[3] - q[2]; e3 = q[0] - q[3]
    lens = [np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)]
    dirs = [e0, e1, e2, e3]
    return _unit(dirs[int(np.argmax(lens))])

def resolve_heading_by_largest_cc(bin_mask, obb_center, quad_pts,
                                  min_area=30, wake_is_behind=True, debug=False):
    """
    用“最大连通区域的质心”相对 OBB 中心，在长边轴的投影正负决定朝向：
      - 航向轴与 OBB 长边平行（u）
      - 取最大 CC 的质心 c_cc，与船中心 c_ship 的向量 d = c_cc - c_ship
      - s = d·u
        * 若 wake_is_behind=True（尾迹在后）：希望最大CC在“后方”(s<0)
          - 若 s>=0，说明当前 u 指向了尾迹 -> 翻转 v=-u
          - 若 s<0，保持 v= u
        * 若 wake_is_behind=False（尾迹在前）：相反判定
    返回：heading_deg(float), v_img(np.array([vx,vy]))；若无有效CC返回(None, None)
    """
    H, W = bin_mask.shape[:2]
    # 取最大 CC
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    max_id, max_area_val = -1, -1
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area and area > max_area_val:
            max_area_val = area
            max_id = i
    if max_id < 0:
        if debug: print("[largest-CC] no valid CC")
        return None, None

    cc_cx, cc_cy = float(cents[max_id, 0]), float(cents[max_id, 1])
    ship_cx, ship_cy = float(obb_center[0]), float(obb_center[1])

    # OBB 长边方向（无向）
    u = _long_edge_unit_from_quad(quad_pts)           # 平行长边
    d = np.array([cc_cx - ship_cx, cc_cy - ship_cy], dtype=np.float64)
    s = float(d[0]*u[0] + d[1]*u[1])                  # 最大CC质心在长边轴的投影

    # 决定朝向
    if wake_is_behind:
        v = u if s < 0 else -u
    else:
        v = u if s >= 0 else -u

    heading_deg = vec_to_compass_deg(v[0], v[1])

    if debug:
        where = "behind(-)" if s < 0 else "ahead(+)"
        print(f"[largest-CC] area={max_area_val}  s={s:.1f} ({where})  heading={heading_deg:.2f}°")

    return float(heading_deg), np.array([float(v[0]), float(v[1])], dtype=np.float32)


# def overlay_seg_mask(model, rgb_np, bgr_np, thr=0.2, alpha=0.4):
#     import torch.nn.functional as F
#     H, W = rgb_np.shape[:2]
#
#     # 归一化 -> Tensor
#     im = rgb_np.astype(np.float32) / 255.0
#     mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#     std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#     im = (im - mean) / std
#     im_t = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()
#     if next(model.parameters()).is_cuda:
#         im_t = im_t.cuda()
#
#     stats = {"pos_px": 0, "area_ratio": 0.0, "num_regions": 0, "max_area": 0}
#     with torch.no_grad():
#         outs = model(im_t, return_seg=True)         # 你模型里已经支持这个
#         seg_logits = outs['seg_logits']             # (1,1,h',w')
#         seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)
#         prob = torch.sigmoid(seg_logits)[0, 0].float().cpu().numpy()
#         bin_mask = (prob > thr).astype(np.uint8)    # (H,W) {0,1}
#
#     # 统计：像素数、占比、连通域数、最大连通域面积
#     pos_px = int(bin_mask.sum())
#     stats["pos_px"] = pos_px
#     stats["area_ratio"] = float(pos_px) / float(H * W)
#     # 连通域
#     num_cc, labels, cc_stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
#     stats["num_regions"] = int(max(0, num_cc - 1))
#     if stats["num_regions"] > 0:
#         areas = cc_stats[1:, cv2.CC_STAT_AREA]      # 排除背景
#         stats["max_area"] = int(areas.max())
#
#     # 可视化叠加
#     color_layer = np.zeros_like(bgr_np, dtype=np.uint8)
#     color_layer[bin_mask == 1] = (0, 0, 255)        # BGR 红
#     vis = cv2.addWeighted(bgr_np, 1.0, color_layer, alpha, 0.0)
#     return vis, bin_mask, stats


def overlay_seg_mask(model, rgb_np, bgr_np, thr=0.6, alpha=0.4, img_size=640):
    H0, W0 = rgb_np.shape[:2]

    rgb_lb, _, ratio, pads = letterbox_pair(rgb_np, None, new_shape=img_size)

    im = rgb_lb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    im = (im - mean) / std
    im_t = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).float()
    if next(model.parameters()).is_cuda:
        im_t = im_t.cuda()

    with torch.no_grad():
        outs = model(im_t, return_seg=True)
        seg_logits = outs['seg_logits']              # (1,1,H_lb,W_lb)
        prob_lb = torch.sigmoid(seg_logits)[0,0].float().cpu().numpy()

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
    num_colors = {'VOC' : 20 ,
            'IC15': 1,
            'IC13': 1,
            'HRSC2016': 1,
            'DOTA':15,
            'UCAS_AOD':2,
            'NWPU_VHR':10
            }
    if num_colors[dataset] == 1:
        colors = [(0, 255, 0)]
    elif num_colors[dataset] == 2:
        colors = [(0, 255, 0), (0, 0, 255)]
    else:
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(num_colors[dataset])]
    return colors


def demo(args):
    hyps = hyp_parse(args.hyp)
    ds = DATASETS[args.dataset](level = 1)
    model = RetinaNet(backbone=args.backbone, hyps=hyps)
    colors = generate_colors(args.dataset)
    if args.weight.endswith('.pth'):
        chkpt = torch.load(args.weight)
        # load model
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        print('load weight from: {}'.format(args.weight))
    model.eval()

    t0 = time.time()
    if not args.dataset == 'DOTA':
        ims_list = [x for x in os.listdir(args.ims_dir) if is_image(x)]

        out_vis_dir = os.path.join('outputs', '1222')
        os.makedirs(out_vis_dir, exist_ok=True)

        out_seg_dir = os.path.join(out_vis_dir, 'seg_masks')
        if args.save_seg:
            os.makedirs(out_seg_dir, exist_ok=True)

        for idx, im_name in enumerate(ims_list):
            s = ''
            t = time.time()
            im_path = os.path.join(args.ims_dir, im_name)   
            s += 'image %g/%g %s: ' % (idx, len(ims_list), im_path)

            # src = cv2.imread(im_path, cv2.IMREAD_COLOR)
            # im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

            src = cv2.imread(im_path, cv2.IMREAD_COLOR)  # 原图(BGR)用于可视化
            im_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)  # 检测用的 RGB

            # 保存一份原图，用于稍后叠加彩色连通域
            bgr0 = src.copy()

            # 分割专用灰度 → 3 通道（模型输入仍是3通道）
            if args.demo_gray:
                gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                seg_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # 分割用的“灰度伪RGB”
            else:
                seg_rgb = im_rgb  # 需要时也可用彩色做分割

            # === 分割（灰度） ===
            # src, bin_mask, seg_stats = overlay_seg_mask(
            #     model, seg_rgb, src,
            #     thr=0.6, alpha=0.4, img_size=args.target_size[0]
            # )

            bin_mask = None
            seg_stats = {"pos_px": 0, "area_ratio": 0.0, "num_regions": 0, "max_area": 0}

            if args.show_seg or args.save_seg:
                src_vis, bin_mask, seg_stats = overlay_seg_mask(
                    model, seg_rgb, src,
                    thr=0.5, alpha=0.4, img_size=args.target_size[0]
                )
                if args.show_seg:
                    src = src_vis  # 只有 show_seg 才把叠加结果显示到 src 上

            # # 用多色连通域覆盖原本的红色掩膜可视化
            # src = overlay_ccs_on_image(bgr0, bin_mask, alpha=0.45, min_area=20, draw_ids=True)

            # === 检测（彩色） ===
            cls_dets = im_detect(model, im_rgb, target_sizes=args.target_size)


            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                bbox = cls_dets[j, 2:]
                if len(bbox) == 4:
                    if args.show_det_label:
                        draw_caption(src, bbox, '{:1.3f}'.format(scores))
                    cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=4)
                else:
                    pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, thickness=3, color=colors[int(cls-1)])

                    put_label = args.show_det_label

                    plot_anchor = False
                    if put_label:
                        label = ds.return_class(cls) + str(' %.2f' % scores)
                        fontScale = 0.45
                        font = cv2.FONT_HERSHEY_COMPLEX
                        thickness = 1
                        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                        c1 = tuple(bbox[:2].astype('int'))
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                        # import ipdb;ipdb.set_trace()

                        cv2.rectangle(src, c1, c2, colors[int(cls-1)], -1)  # filled
                        cv2.putText(src, label, (c1[0], c1[1] -4), font, fontScale, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)
                        if plot_anchor:
                            pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                            cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)


            # print('%sDone. (%.3fs) %d objs' % (s, time.time() - t, len(cls_dets)))

            # # === 从OBB中选一个代表航向的角度/方向（由quad反推，保证与框一致） ===
            # heading_rad_raw = None  # 模型原始弧度（仅用于打印对比）
            # best_area = -1.0
            # best_quad = None
            # best_center = None
            #
            # for j in range(len(cls_dets)):
            #     bbox = cls_dets[j, 2:]
            #     if len(bbox) >= 5:
            #         cx, cy, w, h, ang = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])
            #         area = w * h
            #         if area > best_area:
            #             best_area = area
            #             heading_rad_raw = ang  # 仅打印用
            #             quad = rbox_2_quad(bbox[:5]).reshape((4, 2))
            #             best_quad = quad
            #             best_center = (cx, cy)
            #
            # if best_quad is not None:
            #     # 1) 用quad计算罗盘航向（与框长边一致）
            #     heading_deg, v_img = heading_from_quad(best_quad)
            #
            #     # 先存一个回退（四点法）
            #     heading_deg_quad, v_img_quad = heading_deg, v_img
            #
            #     # 2) 打印原始弧度 和 转换后的罗盘角
            #     print(
            #         f"    [DEBUG] raw heading_rad(model) = {heading_rad_raw:.6f} rad  ({heading_rad_raw / np.pi:.3f} π)" if heading_rad_raw is not None else
            #         "    [DEBUG] raw heading_rad(model) = None")
            #     print(f"    [DEBUG] heading_deg(from quad) = {heading_deg:.2f}°")
            #
            #     # 用“长边平行 + 连通域判定朝向”的方法，得到最终显示航向
            #     # 用“最大连通域 + OBB中心 + 长边平行”判定最终显示航向
            #     cc_heading_deg, cc_v_img = resolve_heading_by_first_last(
            #         bin_mask=bin_mask,
            #         obb_center=best_center,
            #         quad_pts=best_quad,
            #         min_area=100,
            #         debug=True  # 调试期先开
            #     )
            #
            #     if cc_heading_deg is not None:
            #         heading_deg, v_img = cc_heading_deg, cc_v_img
            #     else:
            #         heading_deg, v_img = heading_deg_quad, v_img_quad
            #
            #     print(f"    [CC-longedge] display heading = {heading_deg:.2f}°")
            #
            #     # 3) 用（图坐标方向）估计横波波长（按像素）——这里仍用“方向向量”的弧度版
            #     #    把quad方向向量转成弧度给 wavelength 函数，以保持投影方向一致
            #     heading_rad_from_quad = np.arctan2(v_img[1], v_img[0])  # 注意：这是图像坐标下的atan2(y, x)
            #
            #     wl_info = estimate_wavelength_from_mask(bin_mask, heading_rad_from_quad, min_area=20)
            #     lam_px = wl_info["lambda_px"]
            #     n_crests = wl_info["n_crests"]
            #
            #     if args.show_waveinfo:
            #         # 文本
            #         txt = [f"heading={heading_deg:.1f}°"]
            #         txt.append(f"lamda={lam_px:.1f}px" if lam_px is not None else "λ=NA")
            #         txt.append(f"crests={n_crests}")
            #         cv2.putText(src, " / ".join(txt), (12, 26),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            #
            #         # 箭头：从框中心出发更直观
            #         # if best_center is not None:
            #         #     draw_heading_arrow(src, heading_deg, anchor=best_center, length_ratio=0.15, color=(0, 0, 255),
            #         #                        thickness=3)
            #         # else:
            #         #     draw_heading_arrow(src, heading_deg, length_ratio=0.12, color=(0, 0, 255), thickness=3)
            #
            #         # 控制台汇总
            #         print(
            #             f"    Heading={heading_deg:.1f}angle  | lamda={lam_px if lam_px is not None else 'NA'} px | crests={n_crests}")
            # else:
            #     if args.show_waveinfo:
            #         print("   No OBB found -> skip heading/wavelength.")

            print('%sDone. (%.3fs) %d objs | seg: area=%d px (%.1f%%), blobs=%d, max_blob=%d px'
                  % (s, time.time() - t, len(cls_dets),
                     seg_stats["pos_px"],
                     100.0 * seg_stats["area_ratio"],
                     seg_stats["num_regions"],
                     seg_stats["max_area"]))  # area:这一张图被分出来的前景像素总数； blobs：连通域个数； max_blob：最大连通域的像素面积

            # save image

            # 1) 保存可视化结果图（必须是文件路径）
            out_path = os.path.join(out_vis_dir, os.path.basename(im_path))
            cv2.imwrite(out_path, src)

            # 2) 可选：保存分割二值mask（0/255）
            if args.save_seg and (bin_mask is not None):
                seg_name = os.path.splitext(os.path.basename(im_path))[0] + "_mask.png"
                seg_path = os.path.join(out_seg_dir, seg_name)
                cv2.imwrite(seg_path, (bin_mask * 255).astype(np.uint8))

    ## DOTA detct on large image
    else:
        evaluate(args.target_size,
                args.ims_dir,    
                'DOTA',
                args.backbone,
                args.weight,
                hyps = hyps,
                conf = 0.05)
        if  os.path.exists('outputs/dota_out'):
            shutil.rmtree('outputs/dota_out')
        os.mkdir('outputs/dota_out')
        exec('cd outputs &&  rm -rf detections && rm -rf integrated  && rm -rf merged')
        ResultMerge('outputs/detections', 
                    'outputs/integrated',
                    'outputs/merged',
                    'outputs/dota_out')
        img_path = os.path.join(args.ims_dir,'images')
        label_path = 'outputs/dota_out'
        save_imgs =  False
        if save_imgs:
            show_dota_results(img_path,label_path)
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='fca101')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='weights/best-1217.pth')
    # HRSC
    parser.add_argument('--dataset', type=str, default='HRSC2016')
    parser.add_argument('--ims_dir', type=str, default='/home/lab/libr/obb-RetinaNet/wave_dataset/test/Fig-test')  # Sentinel-2-0909re1  Sentinel-2-test

    parser.add_argument('--show_waveinfo', default=False, help='是否在结果图和控制台输出中显示航向角与横波波长信息')
    parser.add_argument('--pixel_size', type=float, default=-1.0, help='以米为单位的像素尺寸（m/px）。若>0则把像素波长换算为米，并估计对水速度。')

    parser.add_argument('--show_seg', default=False,
                        help='whether to overlay segmentation mask on the output image')

    parser.add_argument('--save_seg', default=False,
                        help='whether to save binary segmentation mask (0/255)')

    parser.add_argument('--seg_thr', type=float, default=0.3,
                        help='threshold for segmentation probability')

    parser.add_argument('--seg_alpha', type=float, default=0.8,
                        help='alpha for segmentation overlay')

    parser.add_argument('--show_det_label', default=False,
                        help='whether to draw class name and confidence score')

    parser.add_argument('--target_size', nargs='+', type=int, default=[640])


    parser.add_argument('--demo_gray', default=True,
                        help='use grayscale for inference but overlay on original color image')
    demo(parser.parse_args())

