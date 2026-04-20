
# -*- coding: utf-8 -*-
import os
import math
import cv2
import numpy as np
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET

from utils.augment import *   # 包含 Augment / mask_valid_boxes 等
from utils.utils import plot_gt
from utils.bbox import quad_2_rbox, constraint_theta
from utils.letterbox import letterbox_pair


class HRSCDataset(data.Dataset):

    def __init__(self,
                 dataset=None,        # txt：每行一个“图像绝对路径”
                 augment=False,
                 level=1,
                 fix_neg_offset=False,   # 若负角度整体少 90°，设为 True
                 neg_offset_deg=90.0,
                 # === 新增：mask 相关 ===
                 mask_root='/home/lab/libr/obb-RetinaNet/wave_dataset/Segmentation_masks',  # 分割标注根目录（可选）。若为 None，按图像路径自动推断
                 mask_suffixes=('.png', '.jpg', '.bmp', '.tiff', '.tif'),
                 mask_binary=True,  # True: 返回二值 mask；False: 保留多类/原值
                 mask_positive_values=(255,),  # 二值化时视为正类的像素值集合
                 use_gray=False,):   # 负角度补偿量（默认 +90°）

        self.image_set_path = dataset
        self.image_list = self._load_image_names() if dataset is not None else []
        self.level = level
        self.augment = augment

        self.use_gray = bool(use_gray)

        self.mask_root = mask_root
        self.mask_suffixes = mask_suffixes
        self.mask_binary = bool(mask_binary)
        self.mask_positive_values = set(mask_positive_values)
        # 告警只打一遍
        self._warn_mask_aug_once = False

        # 类别定义（按需修改）
        if self.level == 1:
            self.classes = ('__background__', 'wake')
        elif self.level == 2:
            self.classes = ('__background__', 'ship', 'air.', 'war.', 'mer.')
        elif self.level == 3:
            self.classes = ('__background__', 'ship', 'air.', 'war.', 'mer.', 'Nim.',
                            'Ent.', 'Arl.', 'Whi.', 'Per.', 'San.', 'Tic.', 'Aus.',
                            'Tar.', 'Con.', 'Com.A', 'Car.A', 'Con.A', 'Med.', 'Car.B')
        else:
            raise ValueError(f'Unsupported level: {self.level}')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        # 角度修正选项（仅当你的数据“负角度整体少 90°”时开启）
        self.fix_neg_offset = bool(fix_neg_offset)
        self.neg_offset_deg = float(neg_offset_deg)

    # ------------------------- 基础接口 -------------------------

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_path = self.image_list[index]
        im_bgr = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise FileNotFoundError(f'Image not found or cannot be read: {im_path}')
        im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        # mask新加入
        H, W = im.shape[:2]

        # roidb['boxes']:(N,5)[cx,cy,w,h,a_deg]  roidb['gt_classes']:(N,)
        roidb = self._load_annotation(im_path)
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]

        nt = len(roidb['boxes'])
        # gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)

        # 读取 mask（如果不存在则为全 0）  mask新加入 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        seg_mask = self._load_mask(im_path, target_hw=(H, W))  # (H,W) uint8

        seg_mask = (seg_mask > 0).astype(np.uint8)
        if seg_mask.shape[:2] != (H, W):
            seg_mask = cv2.resize(seg_mask, (W, H), interpolation=cv2.INTER_NEAREST)


        if nt:
            bboxes = roidb['boxes'][gt_inds, :]      # (K,5)  [cx,cy,w,h,a_deg]
            classes = roidb['gt_classes'][gt_inds]   # (K,)

            # 数据增强（角度单位：度）。你的 Augment 实现若假设弧度，请相应调整。
            if self.augment:

                if not hasattr(self, '_warn_mask_aug_once'):
                    self._warn_mask_aug_once = False

                transform = Augment([
                    HSV(0.5, 0.5, p=0.5),
                    HorizontalFlip(0.5),
                    VerticalFlip(0.5),
                    Affine(degree=10, translate=0.1, scale=0.1, p=0.5),
                ], box_mode='xywha')
                # im, bboxes = transform(im, bboxes)

                # # 优先尝试 transform 支持 mask 的调用约定：transform(im, bboxes, mask=mask)
                # try:
                #     im, bboxes, seg_mask = transform(im, bboxes, seg_mask=seg_mask)
                # except TypeError:
                #     # Augment 不支持 mask：只增强图像/框，mask 原样保留
                #     im, bboxes = transform(im, bboxes)
                #     if not self._warn_mask_aug_once:
                #         print('[WARN] Augment does not support `mask=`; mask will not be geometrically transformed.')
                #         self._warn_mask_aug_once = True
                # 尝试常见的 mask 关键字：mask / masks / seg / seg_mask
                applied = False
                for k in ('mask', 'masks', 'seg', 'seg_mask'):
                    try:
                        im, bboxes, seg_mask = transform(im, bboxes, **{k: seg_mask})
                        applied = True
                        break
                    except TypeError:
                        continue

                if not applied:
                    # 实在不支持 mask：只增强图像/框，并给一次性警告
                    im, bboxes = transform(im, bboxes)
                    if not self._warn_mask_aug_once:
                        print('[WARN] Augment(...) 不接受 mask 参数，掩膜将不会做几何增广（可能造成对不齐）。')
                        self._warn_mask_aug_once = True

            # # 可选：负角度整体+90°补偿（只对 <0° 的样本生效）
            # if self.fix_neg_offset and bboxes.size:
            #     neg = bboxes[:, 4] < 0.0
            #     bboxes[neg, 4] += self.neg_offset_deg

            # 角度仅规范到 (-180°, 180°]，不交换 w/h
            if bboxes.size:
                bboxes[:, 4] = self._wrap_to_180(bboxes[:, 4])
                # 保证 w,h > 0（不交换）
                bboxes[:, 2] = np.abs(bboxes[:, 2])
                bboxes[:, 3] = np.abs(bboxes[:, 3])

            if hasattr(self, 'img_size') and self.img_size:
                im, seg_mask, (r, _), (dw, dh) = letterbox_pair(im, seg_mask, new_shape=self.img_size)
                if bboxes.size:
                    bboxes[:, 0] = bboxes[:, 0] * r + dw  # cx
                    bboxes[:, 1] = bboxes[:, 1] * r + dh  # cy
                    bboxes[:, 2] = bboxes[:, 2] * r  # w
                    bboxes[:, 3] = bboxes[:, 3] * r  # h

            # 写入 [cx,cy,w,h,a_deg]（前5列），第6列后面写 cls_id
            # gt_boxes[:, :-1] = bboxes

            # 合法性过滤（去掉异常框）
            keep = mask_valid_boxes(bboxes, return_mask=True)  #  曾经 这里是 mask = mask_valid_boxes
            bboxes = bboxes[keep]
            classes = classes[keep]
            # gt_boxes = gt_boxes[keep]

            gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
            gt_boxes[:, :-1] = bboxes

            # 写入类别 id
            for i in range(len(bboxes)):
                gt_boxes[i, 5] = classes[i]

            # 追加一份 xyxy 覆盖前 4 列，角度仍在第 5 列
            if len(bboxes):
                cx, cy, w, h, a = [bboxes[:, x] for x in range(5)]
                x1 = cx - 0.5 * w
                x2 = cx + 0.5 * w
                y1 = cy - 0.5 * h
                y2 = cy + 0.5 * h
                gt_boxes[:, 0] = x1
                gt_boxes[:, 1] = y1
                gt_boxes[:, 2] = x2
                gt_boxes[:, 3] = y2
                # 第 4 列角度保持为 (-180, 180] 范围内的度

            # 可选：可视化检查
            # plot_gt(im, gt_boxes[:, :5], im_path, mode='xyxya')

        H, W = im.shape[:2]
        seg_mask = (seg_mask > 0).astype(np.uint8)
        if seg_mask.shape[:2] != (H, W):
            seg_mask = cv2.resize(seg_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        return {'image': im, 'boxes': gt_boxes, 'mask': seg_mask, 'path': im_path}

    # ------------------------- 工具与解析 -------------------------

    @staticmethod
    def _wrap_to_180(a_deg_array):
        """
        把角度(度)规范到 (-180, 180]，不交换 w/h。
        支持 numpy 向量或标量（外部已保证是数组）
        """
        return ((a_deg_array + 180.0) % 360.0) - 180.0

    def _load_image_names(self):
        """读取 txt 文件中的图像绝对路径列表"""
        image_set_file = self.image_set_path
        assert os.path.exists(image_set_file), f'Path does not exist: {image_set_file}'
        with open(image_set_file, 'r', encoding='utf-8') as f:
            image_list = [x.strip() for x in f.readlines() if x.strip()]
        return image_list

    def _load_annotation(self, img_path):
        """
        返回：
            {
              'boxes': np.ndarray, shape=(N,5),  [cx,cy,w,h,angle_deg]
              'gt_classes': np.ndarray, shape=(N,)
            }
        """
        xml_path = self._find_xml_path_from_img_path(img_path)
        if not os.path.exists(xml_path):
            # 找不到标注 → 作为负样本
            return {'boxes': np.zeros((0, 5), dtype=np.float32),
                    'gt_classes': np.zeros((0,), dtype=np.int64)}

        with open(xml_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()

        if '<HRSC_Object>' in content:
            return self._parse_hrsc_xml(content)

        if '<robndbox>' in content:
            return self._parse_voc_obb_xml(content)

        # 其他：无目标
        return {'boxes': np.zeros((0, 5), dtype=np.float32),
                'gt_classes': np.zeros((0,), dtype=np.int64)}

    def _find_xml_path_from_img_path(self, img_path):
        """
        路径推断规则：
        1) AllImages/JPEGImages/images/Image/Imgs → Annotations
        2) 同级：同名 .xml
        3) 上一级 Annotations/同名.xml
        """
        root_dir, img_name = os.path.split(img_path)
        base, _ = os.path.splitext(img_name)

        cand = root_dir
        for k in ['AllImages', 'JPEGImages', 'images', 'Image', 'Imgs']:
            if k in cand:
                cand = cand.replace(k, 'Annotations')
                break
        xml1 = os.path.join(cand, base + '.xml')
        if os.path.exists(xml1):
            return xml1

        xml2 = os.path.join(root_dir, base + '.xml')
        if os.path.exists(xml2):
            return xml2

        xml3 = os.path.join(os.path.dirname(root_dir), 'Annotations', base + '.xml')
        if os.path.exists(xml3):
            return xml3

        # 找不到就返回首选路径
        return xml1

    def _parse_hrsc_xml(self, content):
        """
        HRSC 解析：弧度→度，然后返回 [cx,cy,w,h,a_deg] 与类别
        """
        boxes, gt_classes = [], []
        parts = content.split('<HRSC_Object>')
        if parts:
            _ = parts.pop(0)
        for obj in parts:
            if not obj.strip():
                continue
            cls_id = self._sub(obj, '<Class_ID>', '</Class_ID>')
            diff = self._sub(obj, '<difficult>', '</difficult>')
            if cls_id in ['100000027', '100000022'] or diff == '1':
                continue

            cx = float(self._sub(obj, '<mbox_cx>', '</mbox_cx>'))
            cy = float(self._sub(obj, '<mbox_cy>', '</mbox_cy>'))
            w  = float(self._sub(obj, '<mbox_w>',  '</mbox_w>'))
            h  = float(self._sub(obj, '<mbox_h>',  '</mbox_h>'))
            ang_rad = float(self._sub(obj, '<mbox_ang>', '</mbox_ang>'))
            a_deg = ang_rad * 180.0 / math.pi

            boxes.append([cx, cy, w, h, a_deg])
            gt_classes.append(self.class_mapping(cls_id, self.level))

        if len(boxes) == 0:
            return {'boxes': np.zeros((0, 5), dtype=np.float32),
                    'gt_classes': np.zeros((0,), dtype=np.int64)}
        return {'boxes': np.array(boxes, dtype=np.float32),
                'gt_classes': np.array(gt_classes, dtype=np.int64)}

    def _parse_voc_obb_xml(self, content):
        """
        VOC-OBB 解析：弧度→度，然后返回 [cx,cy,w,h,a_deg] 与类别
        """
        boxes, gt_classes = [], []
        root = ET.fromstring(content)

        for obj in root.findall('object'):
            diff = obj.findtext('difficult')
            if diff is not None and diff.strip() == '1':
                continue

            name = obj.findtext('name')
            if not name:
                continue
            rbox = obj.find('robndbox')
            if rbox is None:
                continue

            try:
                cx = float(rbox.findtext('cx'))
                cy = float(rbox.findtext('cy'))
                w  = float(rbox.findtext('w'))
                h  = float(rbox.findtext('h'))
                ang = float(rbox.findtext('angle'))  # 弧度
            except Exception:
                continue

            a_deg = ang * 180.0 / math.pi

            if name not in self.class_to_ind:
                # 未知类别，跳过（也可以 raise）
                continue
            cls_id = self.class_to_ind[name]

            boxes.append([cx, cy, w, h, a_deg])
            gt_classes.append(cls_id)

        if len(boxes) == 0:
            return {'boxes': np.zeros((0, 5), dtype=np.float32),
                    'gt_classes': np.zeros((0,), dtype=np.int64)}
        return {'boxes': np.array(boxes, dtype=np.float32),
                'gt_classes': np.array(gt_classes, dtype=np.int64)}

    def _load_mask(self, img_path, target_hw=None):
        """
        读取与 img_path 对应的分割 mask。
        返回: (H, W) uint8
          - 若 self.mask_binary=True：输出 {0,1}
          - 若 False：保留原值（多类语义分割）
        """
        mask_path = self._find_mask_path_from_img_path(img_path)
        H, W = target_hw if target_hw is not None else (None, None)

        if (mask_path is None) or (not os.path.exists(mask_path)):
            # 不存在 mask -> 全零
            if H is None or W is None:
                # 没有图像尺寸信息时，尝试从图像读
                im = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if im is None:
                    raise FileNotFoundError(f'Image not found: {img_path}')
                H, W = im.shape[:2]
            return np.zeros((H, W), dtype=np.uint8)

        # 读 mask，优先保持标签值不被阉割
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f'Cannot read mask: {mask_path}')

        # 转单通道
        if m.ndim == 3:
            # 如果是彩色调色板/灰度存 PNG，这里统一取灰度
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)

        # 尺寸对齐
        if (H is not None) and (W is not None) and (m.shape[0] != H or m.shape[1] != W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        # 二值化（可选）
        if self.mask_binary:
            # 把在 positive 集合内的像素归为 1，其余为 0
            pos = np.zeros_like(m, dtype=bool)
            for v in self.mask_positive_values:
                pos |= (m == v)
            m = pos.astype(np.uint8)  # {0,1}

        return m

    def _find_mask_path_from_img_path(self, img_path):
        root_dir, img_name = os.path.split(img_path)
        base, _ = os.path.splitext(img_name)

        # 0) 若指定了 mask_root，优先：mask_root/同名+后缀
        if self.mask_root is not None:
            for suf in self.mask_suffixes:
                p = os.path.join(self.mask_root, base + suf)
                if os.path.exists(p):
                    return p

        # 1) 替换式推断 Segmentation_masks
        cand_dir = root_dir
        for k in ['AllImages', 'JPEGImages', 'images', 'Image', 'Imgs']:
            if k in cand_dir:
                cand_dir = cand_dir.replace(k, 'Segmentation_masks')
                break
        for suf in self.mask_suffixes:
            p = os.path.join(cand_dir, base + suf)
            if os.path.exists(p):
                return p

        # 2) VOC 常见命名 SegmentationClass
        cand_dir2 = root_dir
        for k in ['AllImages', 'JPEGImages', 'images', 'Image', 'Imgs']:
            if k in cand_dir2:
                cand_dir2 = cand_dir2.replace(k, 'SegmentationClass')
                break
        for suf in self.mask_suffixes:
            p = os.path.join(cand_dir2, base + suf)
            if os.path.exists(p):
                return p

        # 3) 同级目录
        for suf in self.mask_suffixes:
            p = os.path.join(root_dir, base + suf)
            if os.path.exists(p):
                return p

        return None

    # ------------------------- 类别映射 -------------------------

    @staticmethod
    def _sub(s, l, r):
        i = s.find(l)
        if i < 0:
            return ''
        i += len(l)
        j = s.find(r, i)
        if j < 0:
            return ''
        return s[i:j]

    def class_mapping(self, cls_id, level):
        if level == 1:
            return 1  # 单类

        if level == 2:
            if cls_id in ['100000005', '100000006', '100000012', '100000013', '100000016', '10000032']:
                cls_id = '100000002'
            if cls_id in ['100000007', '100000008', '100000009', '100000010', '100000011', '10000015',
                          '10000017', '10000019', '10000028', '10000029']:
                cls_id = '100000003'
            if cls_id in ['100000018', '100000020', '100000024', '100000025', '100000026', '10000030']:
                cls_id = '100000004'
            class_ID = ['bg', '100000001', '100000002', '100000003', '100000004']
            return class_ID.index(cls_id)

        if level == 3:
            if cls_id in ['1000000012', '1000000013', '1000000032']:
                cls_id = '100000002'
            if cls_id in ['100000017', '100000028']:
                cls_id = '100000003'
            if cls_id in ['100000024', '100000026']:
                cls_id = '100000004'
            class_ID = ['bg', '100000001', '100000002', '100000003', '100000004', '100000005',
                        '100000006', '100000007', '100000008', '100000009', '1000000010',
                        '1000000011', '100000015', '1000000016', '100000018', '100000019',
                        '100000020', '100000025', '100000029', '100000030']
            return class_ID.index(cls_id)

        raise ValueError(f'Unsupported level: {level}')

    def return_class(self, id_):
        id_ = int(id_)
        return self.classes[id_]


if __name__ == '__main__':
    pass
