# tools/vis_obb_labels.py
# -*- coding:utf-8 -*-
import os, math, argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def normalize_xywha_deg(b):
    """
    [cx,cy,w,h,a_deg] → WakeNet 风格：
    - angle ∈ (-90, 90]
    - angle 表示长边方向
    - 超界时交换 w/h
    """
    if b.size == 0:
        return b

    b = b.copy()
    a = ((b[:, 4] + 180.0) % 360.0) - 180.0  # (-180,180]

    m1 = a <= -90.0
    if np.any(m1):
        a[m1] += 180.0
        w = b[m1, 2].copy()
        b[m1, 2] = b[m1, 3]
        b[m1, 3] = w

    m2 = a > 90.0
    if np.any(m2):
        a[m2] -= 180.0
        w = b[m2, 2].copy()
        b[m2, 2] = b[m2, 3]
        b[m2, 3] = w

    b[:, 4] = a
    return b


def rbox_deg_to_quad(cx, cy, w, h, a_deg):
    """[cx,cy,w,h,angle_deg] → 4点"""
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    dw, dh = w / 2.0, h / 2.0

    corners = np.array([
        [-dw, -dh],
        [ dw, -dh],
        [ dw,  dh],
        [-dw,  dh]
    ], dtype=np.float32)

    R = np.array([[ca, -sa],
                  [sa,  ca]], dtype=np.float32)
    pts = (corners @ R.T) + np.array([cx, cy], dtype=np.float32)
    return pts.astype(np.int32)


def find_xml(img_path):
    root, name = os.path.split(img_path)
    base, _ = os.path.splitext(name)
    cand = root
    for k in ['AllImages', 'JPEGImages', 'images', 'Image', 'Imgs']:
        if k in cand:
            cand = cand.replace(k, 'Annotations')
            break
    for p in [
        os.path.join(cand, base + '.xml'),
        os.path.join(root, base + '.xml'),
        os.path.join(os.path.dirname(root), 'Annotations', base + '.xml')
    ]:
        if os.path.exists(p):
            return p
    return None


def parse_xml(xml_path):
    if xml_path is None or not os.path.exists(xml_path):
        return np.zeros((0,5), np.float32), []

    with open(xml_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    boxes, names = [], []

    # HRSC
    if '<HRSC_Object>' in content:
        parts = content.split('<HRSC_Object>')
        _ = parts.pop(0)
        for obj in parts:
            if not obj.strip(): continue
            diff = _sub(obj, '<difficult>', '</difficult>')
            if diff == '1': continue
            cx = float(_sub(obj, '<mbox_cx>', '</mbox_cx>'))
            cy = float(_sub(obj, '<mbox_cy>', '</mbox_cy>'))
            w  = float(_sub(obj, '<mbox_w>',  '</mbox_w>'))
            h  = float(_sub(obj, '<mbox_h>',  '</mbox_h>'))
            ang_rad = float(_sub(obj, '<mbox_ang>', '</mbox_ang>'))
            a_deg = ang_rad * 180.0 / math.pi
            boxes.append([cx, cy, w, h, a_deg])
            names.append('ship')

    # VOC-OBB
    elif '<robndbox>' in content:
        root = ET.fromstring(content)
        for o in root.findall('object'):
            diff = o.findtext('difficult')
            if diff is not None and diff.strip() == '1':
                continue
            name = o.findtext('name') or 'obj'
            rbox = o.find('robndbox')
            if rbox is None:
                continue
            cx = float(rbox.findtext('cx'))
            cy = float(rbox.findtext('cy'))
            w  = float(rbox.findtext('w'))
            h  = float(rbox.findtext('h'))
            ang_rad = float(rbox.findtext('angle'))
            a_deg = ang_rad * 180.0 / math.pi
            boxes.append([cx, cy, w, h, a_deg])
            names.append(name)

    if len(boxes) == 0:
        return np.zeros((0,5), np.float32), []

    boxes = np.array(boxes, dtype=np.float32)
    boxes = normalize_xywha_deg(boxes)   # ★关键：和训练一致
    return boxes, names


def _sub(s, l, r):
    i = s.find(l)
    if i < 0: return ''
    i += len(l)
    j = s.find(r, i)
    if j < 0: return ''
    return s[i:j]


def draw_one(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print('missing image:', img_path)
        return

    boxes, names = parse_xml(find_xml(img_path))

    for i, b in enumerate(boxes):
        cx, cy, w, h, a = b
        quad = rbox_deg_to_quad(cx, cy, w, h, a).reshape(-1,1,2)
        cv2.polylines(img, [quad], True, (0,255,0), 2)
        txt = f'{names[i]} {a:.1f}°'
        x0, y0 = quad[0,0]
        cv2.putText(img, txt, (int(x0), int(y0)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/train.txt',)
    ap.add_argument('--save_dir', default='/home/lab/libr/obb-RetinaNet/outputs/vis_gt2')
    args = ap.parse_args()

    with open(args.list, 'r') as f:
        imgs = [x.strip() for x in f if x.strip()]

    for p in imgs:
        out = os.path.join(args.save_dir, os.path.basename(p))
        draw_one(p, out)
        print('save:', out)


if __name__ == '__main__':
    main()
