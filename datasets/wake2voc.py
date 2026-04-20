import os
import math
import shutil
import xml.etree.ElementTree as ET
from glob import glob
import cv2
import numpy as np
import random

# ========= 配置你的数据根目录 =========
DATA_ROOT    = r"/home/lab/libr/obb-RetinaNet/wave_dataset"   # 改成你的根目录
SRC_IMG_DIRS = ["JPEGImages", "images"]         # 优先使用第一个存在的作为图片源
SRC_ANN_DIRS = ["Annotations", "xml", "label"]  # 优先使用第一个存在的作为XML源

# 目标标准VOC目录
DST_IMG_DIR  = os.path.join(DATA_ROOT, "JPEGImages")
DST_ANN_DIR  = os.path.join(DATA_ROOT, "Annotations")
SET_DIR      = os.path.join(DATA_ROOT, "ImageSets", "Main")

# 角度单位：'radian' 或 'degree'（你的样例看起来是弧度）
ANGLE_UNIT   = "radian"

# 训练/验证拆分比例
TRAIN_RATIO  = 0.8
RANDOM_SEED  = 2024

# ========= 工具函数 =========
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def find_first_existing(subdirs):
    for sd in subdirs:
        p = os.path.join(DATA_ROOT, sd)
        if os.path.isdir(p):
            return p
    return None

def rbox_to_poly(cx, cy, w, h, angle):
    dx, dy = w / 2.0, h / 2.0
    pts = np.array([[-dx, -dy], [ dx, -dy], [ dx,  dy], [-dx,  dy]], dtype=np.float32)
    if ANGLE_UNIT == "degree":
        angle = math.radians(angle)
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    R = np.array([[cos_t, -sin_t],[sin_t, cos_t]], dtype=np.float32)
    rot = pts @ R.T
    rot[:, 0] += cx
    rot[:, 1] += cy
    return rot  # (4,2)

def poly_to_aabb(poly):
    xs = poly[:, 0]
    ys = poly[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

def clip_box(xmin, ymin, xmax, ymax, W, H):
    xmin = max(0.0, min(xmin, W - 1))
    ymin = max(0.0, min(ymin, H - 1))
    xmax = max(0.0, min(xmax, W - 1))
    ymax = max(0.0, min(ymax, H - 1))
    if xmax < xmin: xmax = xmin
    if ymax < ymin: ymax = ymin
    return xmin, ymin, xmax, ymax

def get_image_size(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h

def set_or_make(parent, tag, text=None):
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    if text is not None:
        node.text = str(text)
    return node

def write_voc_bndbox(obj_node, xmin, ymin, xmax, ymax):
    bnd = obj_node.find('bndbox')
    if bnd is None:
        bnd = ET.SubElement(obj_node, 'bndbox')
    set_or_make(bnd, 'xmin', int(round(xmin)))
    set_or_make(bnd, 'ymin', int(round(ymin)))
    set_or_make(bnd, 'xmax', int(round(xmax)))
    set_or_make(bnd, 'ymax', int(round(ymax)))
    return bnd

# ========= 主流程 =========
def main():
    ensure_dir(DST_IMG_DIR)
    ensure_dir(DST_ANN_DIR)
    ensure_dir(SET_DIR)

    # 找到源图片目录、XML目录
    src_img_dir = find_first_existing(SRC_IMG_DIRS)
    if src_img_dir is None:
        raise FileNotFoundError(f"找不到图片目录（在 {SRC_IMG_DIRS} 中）")
    src_img_dir = os.path.join(DATA_ROOT, src_img_dir)

    src_ann_dir = find_first_existing(SRC_ANN_DIRS)
    if src_ann_dir is None:
        raise FileNotFoundError(f"找不到XML目录（在 {SRC_ANN_DIRS} 中）")
    src_ann_dir = os.path.join(DATA_ROOT, src_ann_dir)

    print(f"[INFO] 图片目录: {src_img_dir}")
    print(f"[INFO] XML目录 : {src_ann_dir}")
    print(f"[INFO] 目标VOC: images->{DST_IMG_DIR}, xml->{DST_ANN_DIR}")

    # 收集全部xml
    xml_paths = sorted(glob(os.path.join(src_ann_dir, "*.xml")))
    print(f"[INFO] 待处理XML数量: {len(xml_paths)}")

    ok_ids = []

    for xml_path in xml_paths:
        base = os.path.splitext(os.path.basename(xml_path))[0]

        # 找对应图片
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = os.path.join(src_img_dir, base + ext)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            # XML里可能写了 filename
            try:
                root_tmp = ET.parse(xml_path).getroot()
                fn = root_tmp.find('filename').text if root_tmp.find('filename') is not None else None
                if fn:
                    cand = os.path.join(src_img_dir, fn)
                    if os.path.exists(cand):
                        img_path = cand
                        base = os.path.splitext(fn)[0]
            except Exception:
                pass

        if img_path is None:
            print(f"[WARN] 图片缺失，跳过: {base}")
            continue

        # 读取图像尺寸
        size = get_image_size(img_path)
        if size is None:
            print(f"[WARN] 打不开图片，跳过: {img_path}")
            continue
        W, H = size

        # 复制图片到 VOC 目录（若源不在 DST_IMG_DIR 中）
        dst_img_path = os.path.join(DST_IMG_DIR, base + ".jpg")
        if os.path.abspath(os.path.dirname(img_path)) != os.path.abspath(DST_IMG_DIR) or not dst_img_path.lower().endswith(os.path.splitext(img_path)[1].lower()):
            # 统一转存为 .jpg
            img = cv2.imread(img_path)
            cv2.imwrite(dst_img_path, img)
        else:
            # 已经是目标目录，且扩展名为 .jpg
            dst_img_path = img_path

        # 解析并写回 VOC bndbox
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 标准化头部信息
        set_or_make(root, 'folder', 'JPEGImages')
        set_or_make(root, 'filename', os.path.basename(dst_img_path))
        set_or_make(root, 'path', dst_img_path)

        size_node = set_or_make(root, 'size')
        set_or_make(size_node, 'width', W)
        set_or_make(size_node, 'height', H)
        set_or_make(size_node, 'depth', 3)  # 若非彩色，可在此改成1

        objects = root.findall('object')
        if len(objects) == 0:
            print(f"[WARN] 无<object>：{xml_path}")
            continue

        valid_obj = 0
        for obj in objects:
            diff = obj.find('difficult')
            if diff is not None:
                try:
                    if int(diff.text) == 1:
                        continue
                except Exception:
                    pass

            # 优先 robndbox
            robnd = obj.find('robndbox')
            if robnd is not None:
                try:
                    cx = float(robnd.find('cx').text)
                    cy = float(robnd.find('cy').text)
                    w  = float(robnd.find('w').text)
                    h  = float(robnd.find('h').text)
                    ang= float(robnd.find('angle').text)
                except Exception as e:
                    print(f"[WARN] robndbox字段异常: {xml_path} | {e}")
                    continue

                poly = rbox_to_poly(cx, cy, w, h, ang)
                xmin, ymin, xmax, ymax = poly_to_aabb(poly)
                xmin, ymin, xmax, ymax = clip_box(xmin, ymin, xmax, ymax, W, H)
                write_voc_bndbox(obj, xmin, ymin, xmax, ymax)
                valid_obj += 1
                continue

            # 兼容已有 bndbox（直接校验裁剪）
            bnd = obj.find('bndbox')
            if bnd is not None:
                try:
                    xmin = float(bnd.find('xmin').text)
                    ymin = float(bnd.find('ymin').text)
                    xmax = float(bnd.find('xmax').text)
                    ymax = float(bnd.find('ymax').text)
                except Exception as e:
                    print(f"[WARN] bndbox字段异常: {xml_path} | {e}")
                    continue
                xmin, ymin, xmax, ymax = clip_box(xmin, ymin, xmax, ymax, W, H)
                write_voc_bndbox(obj, xmin, ymin, xmax, ymax)
                valid_obj += 1
            else:
                # 无robndbox也无bndbox
                continue

        if valid_obj == 0:
            print(f"[WARN] 没有有效目标：{xml_path}")
            continue

        # 保存新XML到 VOC Annotations
        dst_xml_path = os.path.join(DST_ANN_DIR, base + ".xml")
        tree.write(dst_xml_path, encoding='utf-8', xml_declaration=True)
        ok_ids.append(base)

    print(f"[INFO] 成功转换：{len(ok_ids)} 个")

    # 生成 ImageSets 列表
    random.seed(RANDOM_SEED)
    ok_ids = sorted(list(set(ok_ids)))
    random.shuffle(ok_ids)
    n = len(ok_ids)
    n_tr = int(TRAIN_RATIO * n)
    train_ids = ok_ids[:n_tr]
    val_ids   = ok_ids[n_tr:]
    ensure_dir(SET_DIR)
    def write_list(fn, ids):
        with open(os.path.join(SET_DIR, fn), "w", encoding="utf-8") as f:
            for i in ids:
                f.write(i + "\n")
        print(f"[INFO] 写出 {fn}: {len(ids)}")

    write_list("train.txt", train_ids)
    write_list("val.txt",   val_ids)
    write_list("trainval.txt", ok_ids)

    print("[DONE] 转换完成。")

if __name__ == "__main__":
    main()
