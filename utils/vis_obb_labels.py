# tools/vis_obb_labels.py
# -*- coding:utf-8 -*-
import os, math, argparse
import cv2
import numpy as np
import xml.etree.ElementTree as ET


def normalize_xywha_to_pi_deg(b):
    """
    只把角度规范到 (-180°, 180°]，不交换 w/h，也不改其他字段
    """
    if b.size == 0:
        return b
    b = b.copy()
    a = b[:, 4]
    a = ((a + 180.0) % 360.0) - 180.0  # (-180, 180]
    b[:, 4] = a
    return b


def rbox_deg_to_quad(cx, cy, w, h, a_deg):
    """[cx,cy,w,h,angle_deg] -> 4点 (x,y)，顺时针"""
    a = np.deg2rad(a_deg)
    ca, sa = np.cos(a), np.sin(a)
    dw, dh = w / 2.0, h / 2.0
    # 以矩形中心为原点的四角
    corners = np.array([[-dw, -dh],
                        [ dw, -dh],
                        [ dw,  dh],
                        [-dw,  dh]], dtype=np.float32)
    R = np.array([[ca, -sa],
                  [sa,  ca]], dtype=np.float32)
    pts = (corners @ R.T) + np.array([cx, cy], dtype=np.float32)
    return pts.astype(np.int32)

def normalize_xywha_deg(b):
    """
    b: (N,5) -> [cx,cy,w,h,a_deg]
    角度归一化到 (-90, 90]；若超界则 ±180 并交换 w/h
    """
    if b.size == 0:
        return b
    b = b.copy()
    a = b[:, 4]
    a = ((a + 180.0) % 360.0) - 180.0
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
    b[:, 2] = np.abs(b[:, 2])
    b[:, 3] = np.abs(b[:, 3])
    return b

def find_xml(img_path):
    root, name = os.path.split(img_path)
    base, _ = os.path.splitext(name)
    cand = root
    for k in ['AllImages', 'JPEGImages', 'images', 'Image', 'Imgs']:
        if k in cand:
            cand = cand.replace(k, 'Annotations')
            break
    xml1 = os.path.join(cand, base + '.xml')
    if os.path.exists(xml1): return xml1
    xml2 = os.path.join(root, base + '.xml')
    if os.path.exists(xml2): return xml2
    xml3 = os.path.join(os.path.dirname(root), 'Annotations', base + '.xml')
    if os.path.exists(xml3): return xml3
    return None

def parse_xml(xml_path):
    """
    返回：boxes (N,5)[cx,cy,w,h,a_deg], names (N,)
    兼容 HRSC 与 VOC-OBB
    """
    if xml_path is None or (not os.path.exists(xml_path)):
        return np.zeros((0,5), dtype=np.float32), []
    with open(xml_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()
    boxes, names = [], []

    # HRSC
    if '<HRSC_Object>' in content:
        parts = content.split('<HRSC_Object>')
        _ = parts.pop(0)
        for obj in parts:
            if not obj.strip(): continue
            difficult = _sub(obj, '<difficult>', '</difficult>')
            if difficult == '1': continue
            cx = float(_sub(obj, '<mbox_cx>', '</mbox_cx>'))
            cy = float(_sub(obj, '<mbox_cy>', '</mbox_cy>'))
            w  = float(_sub(obj, '<mbox_w>',  '</mbox_w>'))
            h  = float(_sub(obj, '<mbox_h>',  '</mbox_h>'))
            ang_rad = float(_sub(obj, '<mbox_ang>', '</mbox_ang>'))
            a_deg = ang_rad * 180.0 / math.pi
            boxes.append([cx, cy, w, h, a_deg])
            names.append('ship')  # HRSC 可自行映射
        if len(boxes)==0:
            return np.zeros((0,5), dtype=np.float32), []
        return np.array(boxes, dtype=np.float32), names

    # VOC-OBB
    if '<robndbox>' in content:
        root = ET.fromstring(content)
        for o in root.findall('object'):
            diff = o.findtext('difficult')
            if diff is not None and diff.strip() == '1': continue
            name = o.findtext('name') or 'obj'
            rbox = o.find('robndbox')
            if rbox is None: continue
            cx = float(rbox.findtext('cx')); cy = float(rbox.findtext('cy'))
            w  = float(rbox.findtext('w'));  h  = float(rbox.findtext('h'))
            ang = float(rbox.findtext('angle'))  # 弧度
            a_deg = ang * 180.0 / math.pi
            if a_deg < 0:
                a_deg += 90.0

            boxes.append([cx, cy, w, h, a_deg]); names.append(name)
        if len(boxes)==0:
            return np.zeros((0,5), dtype=np.float32), []
        return np.array(boxes, dtype=np.float32), names

    return np.zeros((0,5), dtype=np.float32), []

def _sub(s, l, r):
    i = s.find(l);
    if i < 0: return ''
    i += len(l); j = s.find(r, i)
    if j < 0: return ''
    return s[i:j]

def fix_neg_deg_inplace(deg_array):
    """只修正<0°的角度：加90°"""
    if deg_array.size:
        neg = deg_array < 0
        deg_array[neg] += 90.0
    return deg_array


def draw_one(img_path, save_path, palette=None, put_label=True):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print('missing image:', img_path); return
    h, w = img_bgr.shape[:2]
    boxes, names = parse_xml(find_xml(img_path))
    
    # boxes = normalize_xywha_deg(boxes)
    boxes = normalize_xywha_to_pi_deg(boxes)

    if palette is None:
        palette = {}
    for i, b in enumerate(boxes):
        cx, cy, bw, bh, a = b
        cls = names[i] if i < len(names) else 'obj'
        if cls not in palette:
            palette[cls] = (int(np.random.randint(0,255)),
                            int(np.random.randint(0,255)),
                            int(np.random.randint(0,255)))
        color = palette[cls]
        quad = rbox_deg_to_quad(cx, cy, bw, bh, a).reshape(-1,1,2)
        cv2.polylines(img_bgr, [quad], isClosed=True, color=color, thickness=2)
        if put_label:
            x0, y0 = np.min(quad[:,0,0]), np.min(quad[:,0,1])
            txt = f'{cls} {a:.1f}°'
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x1, y1 = int(x0), int(max(0, y0 - th - 4))
            cv2.rectangle(img_bgr, (x1, y1), (x1+tw+4, y1+th+4), color, -1)
            cv2.putText(img_bgr, txt, (x1+2, y1+th+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    if boxes.shape[0] == 0:
        cv2.putText(img_bgr, 'NO GT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img_bgr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', default='/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/train.txt', help='txt: one absolute image path per line')
    ap.add_argument('--save_dir', default='/home/lab/libr/obb-RetinaNet/outputs/vis_gt1')
    ap.add_argument('--max', type=int, default=0, help='visualize first N images; 0=all')
    args = ap.parse_args()

    with open(args.list, 'r', encoding='utf-8') as f:
        imgs = [x.strip() for x in f if x.strip()]

    if args.max > 0:
        imgs = imgs[:args.max]

    for p in imgs:
        base = os.path.splitext(os.path.basename(p))[0]
        out = os.path.join(args.save_dir, base + '.jpg')
        draw_one(p, out)
        print('save:', out)

if __name__ == '__main__':
    main()
