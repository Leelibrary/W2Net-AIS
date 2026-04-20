# --------------------------------------------------------
# WakeDataset for robndbox XML (cx, cy, w, h, angle)
# Written for "wake" class, extensible to more classes
# --------------------------------------------------------
import os
import math
import cv2
import numpy as np
import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict, Any


def _endswith_any(path: str, suffixes: List[str]) -> Tuple[bool, str]:
    """Return (matched, suffix) if path endswith any suffix in suffixes."""
    for s in suffixes:
        if path.replace("\\", "/").endswith(s.replace("\\", "/")):
            return True, s
    return False, ""


def _infer_data_root_from_set_file(set_file: str) -> str:
    """
    从 ImageSets/Main/*.txt 推回数据根目录（含 JPEGImages 与 Annotations）
    期望结构：
    <root>/
      ├─ JPEGImages/
      ├─ Annotations/
      └─ ImageSets/Main/*.txt
    """
    set_file = os.path.abspath(set_file)
    ok, suf = _endswith_any(set_file, [
        os.path.join("ImageSets", "Main", "train.txt"),
        os.path.join("ImageSets", "Main", "trainval.txt"),
        os.path.join("ImageSets", "Main", "val.txt"),
        os.path.join("ImageSets", "Main", "test.txt"),
    ])
    if ok:
        root = set_file[:-len(suf)]
    else:
        # 兜底：如果用户直接给了根目录，也允许
        root = os.path.dirname(os.path.dirname(set_file))
    return root.rstrip("/\\")  # 去尾部斜杠


def rbox_to_poly(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """
    rbox -> 4点多边形（顺序：左上、右上、右下、左下），返回 [x1,y1,...,x4,y4]
    theta 为相对 x 轴的旋转角（弧度），逆时针为正。
    """
    # 以中心为原点的矩形4点
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([
        [-dx, -dy],  # 左上
        [ dx, -dy],  # 右上
        [ dx,  dy],  # 右下
        [-dx,  dy],  # 左下
    ], dtype=np.float32)

    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]], dtype=np.float32)
    rot = pts @ R.T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot.reshape(-1)  # [x1,y1,x2,y2,x3,y3,x4,y4]


def poly_to_aabb(poly8: np.ndarray) -> Tuple[float, float, float, float]:
    """8维多边形转外接直框 (xmin,ymin,xmax,ymax)"""
    xs = poly8[0::2]
    ys = poly8[1::2]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def normalize_angle(angle: float, unit: str = "radian") -> float:
    """将角度规范到 (-pi, pi] 或 (-180, 180]"""
    if unit == "degree":
        while angle <= -180.0:
            angle += 360.0
        while angle > 180.0:
            angle -= 360.0
        return angle
    else:
        # radian
        two_pi = 2.0 * math.pi
        while angle <= -math.pi:
            angle += two_pi
        while angle > math.pi:
            angle -= two_pi
        return angle


class WakeDataset(data.Dataset):
    """
    读取 <robndbox> 的旋转框数据集。
    默认返回 rbox：(cx, cy, w, h, angle, class_id)，numpy.float32。
    也支持返回 poly8（8点）或 aabb（直框）格式，均在末尾附 class_id。
    """

    def __init__(self,
                 set_file: str,                       # e.g. ".../ImageSets/Main/trainval.txt"
                 classes: Tuple[str, ...] = ('__background__', 'wake'),
                 image_exts: Tuple[str, ...] = ('.jpg', '.png', '.jpeg', '.bmp'),
                 return_format: str = 'rbox',         # 'rbox' | 'poly' | 'aabb'
                 angle_unit: str = 'radian',          # 'radian' | 'degree'
                 random_flip: bool = True):
        """
        参数说明：
        - set_file:     列表txt路径（每行一个图像ID，不带扩展名）
        - classes:      类别元组；默认只有 'wake'
        - image_exts:   允许的图片扩展名
        - return_format:'rbox' 返回 (cx,cy,w,h,angle,cls)
                         'poly' 返回 (x1..y4, cls)
                         'aabb' 返回 (xmin,ymin,xmax,ymax,cls)
        - angle_unit:   你的 XML angle 单位（你示例看起来是弧度）
        - random_flip:  是否随机水平翻转（图像与标注同步变换）
        """
        super().__init__()
        assert return_format in ('rbox', 'poly', 'aabb')
        assert angle_unit in ('radian', 'degree')

        self.set_file = os.path.abspath(set_file)
        self.data_root = _infer_data_root_from_set_file(self.set_file)
        self.image_exts = image_exts
        self.return_format = return_format
        self.angle_unit = angle_unit
        self.random_flip = random_flip

        # 读取 ID 列表
        if not os.path.exists(self.set_file):
            raise FileNotFoundError(f"List file not found: {self.set_file}")
        with open(self.set_file, 'r', encoding='utf-8') as f:
            self.image_ids = [ln.strip() for ln in f if ln.strip()]

        # 类别映射
        self.classes = classes
        self.num_classes = len(self.classes)
        self.class_to_ind = {name: i for i, name in enumerate(self.classes)}

        # 目录检查提示（不强制报错，避免影响自定义结构）
        self.img_dir = os.path.join(self.data_root, 'JPEGImages')
        self.ann_dir = os.path.join(self.data_root, 'Annotations')

    def __len__(self):
        return len(self.image_ids)

    def _find_image_path(self, img_id: str) -> str:
        for ext in self.image_exts:
            cand = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(cand):
                return cand
        # 兜底：有些 XML/列表里 filename 带扩展，尝试原样
        for ext in self.image_exts:
            cand = os.path.join(self.img_dir, img_id + ext.upper())
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"Image not found for id={img_id} under {self.img_dir} with {self.image_exts}")

    def _load_xml(self, img_id: str) -> Dict[str, Any]:
        xml_path = os.path.join(self.ann_dir, img_id + '.xml')
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML not found: {xml_path}")
        tree = ET.parse(xml_path)
        return {'tree': tree, 'root': tree.getroot(), 'path': xml_path}

    def _parse_objects(self, root) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析 <object>，仅接受 <robndbox>；可兼容 name 大小写
        返回：
        - boxes: 根据 return_format 不同而不同的列数（末尾不含 class）
        - labels: [N]
        """
        boxes = []
        labels = []
        for obj in root.findall('object'):
            # 过滤 difficult=1
            diff_tag = obj.find('difficult')
            if diff_tag is not None and diff_tag.text is not None:
                try:
                    if int(diff_tag.text) == 1:
                        continue
                except Exception:
                    pass

            name = obj.find('name').text.lower().strip() if obj.find('name') is not None else 'wake'
            if name not in self.class_to_ind:
                # 未知类别则跳过
                continue
            cls_id = self.class_to_ind[name]

            robnd = obj.find('robndbox')
            bnd = obj.find('bndbox')

            if robnd is not None:
                # 读取旋转框
                cx = float(robnd.find('cx').text)
                cy = float(robnd.find('cy').text)
                w  = float(robnd.find('w').text)
                h  = float(robnd.find('h').text)
                angle = float(robnd.find('angle').text)

                if self.return_format == 'rbox':
                    boxes.append([cx, cy, w, h, angle])
                elif self.return_format == 'poly':
                    poly8 = rbox_to_poly(cx, cy, w, h, angle)
                    boxes.append(poly8.tolist())
                else:
                    # aabb
                    poly8 = rbox_to_poly(cx, cy, w, h, angle)
                    xmin, ymin, xmax, ymax = poly_to_aabb(poly8)
                    boxes.append([xmin, ymin, xmax, ymax])

                labels.append(cls_id)

            elif bnd is not None:
                # 兼容直框 bndbox（如果标注里混用）
                xmin = float(bnd.find('xmin').text)
                ymin = float(bnd.find('ymin').text)
                xmax = float(bnd.find('xmax').text)
                ymax = float(bnd.find('ymax').text)
                if self.return_format == 'rbox':
                    # 直框转 rbox，angle=0
                    cx = (xmin + xmax) / 2.0
                    cy = (ymin + ymax) / 2.0
                    w = abs(xmax - xmin)
                    h = abs(ymax - ymin)
                    angle = 0.0 if self.angle_unit == 'radian' else 0.0
                    boxes.append([cx, cy, w, h, angle])
                elif self.return_format == 'poly':
                    poly8 = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], dtype=np.float32)
                    boxes.append(poly8.tolist())
                else:
                    boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)
            else:
                # 既无 robndbox 也无 bndbox，跳过
                continue

        if len(boxes) == 0:
            return np.empty((0, 5 if self.return_format == 'rbox' else (8 if self.return_format == 'poly' else 4)), dtype=np.float32), \
                   np.empty((0,), dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _hflip_inplace(self, image: np.ndarray, boxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        水平翻转；对 rbox/ poly/ aabb 分别处理
        rbox：cx -> W-1-cx；angle -> pi - angle（度制则 180 - angle），再归一化
        """
        H, W = image.shape[:2]
        image = cv2.flip(image, 1)

        if self.return_format == 'rbox':
            # boxes: [N, 5] -> (cx, cy, w, h, angle)
            if boxes.size > 0:
                boxes[:, 0] = (W - 1) - boxes[:, 0]  # cx
                if self.angle_unit == 'degree':
                    boxes[:, 4] = 180.0 - boxes[:, 4]
                    for i in range(boxes.shape[0]):
                        boxes[i, 4] = normalize_angle(boxes[i, 4], 'degree')
                else:
                    boxes[:, 4] = math.pi - boxes[:, 4]
                    for i in range(boxes.shape[0]):
                        boxes[i, 4] = normalize_angle(boxes[i, 4], 'radian')

        elif self.return_format == 'poly':
            # boxes: [N, 8] -> (x1,y1,...,x4,y4)
            if boxes.size > 0:
                xs = boxes[:, 0::2]
                xs[:] = (W - 1) - xs

        else:
            # aabb: [xmin,ymin,xmax,ymax]
            if boxes.size > 0:
                xmin = (W - 1) - boxes[:, 2]
                xmax = (W - 1) - boxes[:, 0]
                boxes[:, 0] = xmin
                boxes[:, 2] = xmax

        return image, boxes

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img_id = self.image_ids[index]
        img_path = self._find_image_path(img_id)
        xml = self._load_xml(img_id)

        # 读图 -> RGB
        im = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # 解析标注
        boxes, labels = self._parse_objects(xml['root'])

        # 随机水平翻转
        if self.random_flip and np.random.rand() >= 0.5:
            im, boxes = self._hflip_inplace(im, boxes)

        # 统一把类别拼到最后一列
        if boxes.size == 0:
            # 保持输出稳定：空标注
            if self.return_format == 'rbox':
                out = np.empty((0, 6), dtype=np.float32)
            elif self.return_format == 'poly':
                out = np.empty((0, 9), dtype=np.float32)
            else:
                out = np.empty((0, 5), dtype=np.float32)
        else:
            if self.return_format == 'rbox':
                out = np.concatenate([boxes, labels.reshape(-1, 1).astype(np.float32)], axis=1)
            elif self.return_format == 'poly':
                out = np.concatenate([boxes, labels.reshape(-1, 1).astype(np.float32)], axis=1)
            else:
                out = np.concatenate([boxes, labels.reshape(-1, 1).astype(np.float32)], axis=1)

        return {
            'image': im,          # np.uint8, HxWx3, RGB
            'boxes': out,         # np.float32, Nx{6|9|5}
            'img_id': img_id,
            'img_path': img_path
        }


def wake_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    自定义 collate：图片堆叠成 list，标注保持 list（不同图的目标数不同）。
    你也可以在这里把 image 转 tensor 并做归一化。
    """
    images = [b['image'] for b in batch]  # list of HxWx3 (RGB)
    boxes  = [b['boxes'] for b in batch]  # list of (N, K)
    img_ids = [b['img_id'] for b in batch]
    img_paths = [b['img_path'] for b in batch]
    return {
        'images': images,
        'targets': boxes,
        'img_ids': img_ids,
        'img_paths': img_paths
    }


# -------------------------- 用法示例 --------------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 假设你的结构是：
    # <root>/
    #   ├─ JPEGImages/
    #   ├─ Annotations/
    #   └─ ImageSets/Main/trainval.txt
    set_file = "path/to/your/dataset/ImageSets/Main/trainval.txt"

    ds = WakeDataset(
        set_file=set_file,
        classes=('__background__', 'wake'),
        return_format='rbox',      # 也可 'poly' 或 'aabb'
        angle_unit='radian',       # 你的 angle 看起来是弧度
        random_flip=True
    )

    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=wake_collate_fn)

    batch = next(iter(dl))
    print("images:", len(batch['images']))
    print("targets lens:", [t.shape for t in batch['targets']])
    print("first sample id:", batch['img_ids'][0])
