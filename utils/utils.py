import cv2
import os
import math
import random
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as transforms

from utils.bbox import rbox_2_quad


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hyp_parse(hyp_path):
    hyp = {}
    keys = [] 
    with open(hyp_path,'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0 : continue
            v = line.strip().split(':')
            try:
                hyp[v[0]] = float(v[1].strip().split(' ')[0])
            except:
                hyp[v[0]] = eval(v[1].strip().split(' ')[0])
            keys.append(v[0])
        f.close()
    return hyp


def model_info(model, report='summary'):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if report == 'full':
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients' % (len(list(model.parameters())), n_p, n_g))


def curriculum_factor(init, final, step=1, mode='suspend_cosine'):
    if mode == 'cosine':
        sequence = [(0.5 - 0.5 * math.cos(math.pi * i / final)) *  (final - init) + init \
            for i in range(init, final+step, step)]

    elif mode == 'suspend_cosine':
        suspend_ratio = 0.1
        suspend_interval = (final - init)*suspend_ratio
        start = suspend_interval + init if suspend_interval > step else init  
        sequence = [(0.5 - 0.5 * math.cos(math.pi * i / final)) *  (final - init) + init \
             if i>start else init  for i in range(init, final+step, step)]
    # vis 
    import matplotlib.pylab as plt
    import numpy as np
    plt.scatter(np.array([x for x in range(init, final+step, step)]),np.array(sequence)) 
    plt.show()


def plot_gt(img, bboxes, im_path, mode='xyxyxyxy'):
    if not os.path.exists('temp'):
        os.mkdir('temp')
    if mode == 'xywha':
        bboxes = rbox_2_quad(bboxes,mode=mode)
    if mode == 'xyxya':
        bboxes = rbox_2_quad(bboxes,mode=mode)
    for box in bboxes:
        img = cv2.polylines(cv2.UMat(img),[box.reshape(-1,2).astype(np.int32)],True,(0,0,255),2)
        cv2.imwrite(os.path.join('temp','augment_%s' % (os.path.split(im_path)[1])),img)
    print('Check augmentation results in `temp` folder!!!')

if __name__ == '__main__':
    curriculum_factor(836, 6400, 32)


def sort_corners(quads):
    sorted = np.zeros(quads.shape, dtype=np.float32)
    for i, corners in enumerate(quads):
        corners = corners.reshape(4, 2)
        centers = np.mean(corners, axis=0)
        corners = corners - centers
        cosine = corners[:, 0] / np.sqrt(corners[:, 0] ** 2 + corners[:, 1] ** 2)
        cosine = np.minimum(np.maximum(cosine, -1.0), 1.0)
        thetas = np.arccos(cosine) / np.pi * 180.0
        indice = np.where(corners[:, 1] > 0)[0]
        thetas[indice] = 360.0 - thetas[indice]
        corners = corners + centers
        corners = corners[thetas.argsort()[::-1], :]
        corners = corners.reshape(8)
        dx1, dy1 = (corners[4] - corners[0]), (corners[5] - corners[1])
        dx2, dy2 = (corners[6] - corners[2]), (corners[7] - corners[3])
        slope_1 = dy1 / dx1 if dx1 != 0 else np.iinfo(np.int32).max
        slope_2 = dy2 / dx2 if dx2 != 0 else np.iinfo(np.int32).max
        if slope_1 > slope_2:
            if corners[0] < corners[4]:
                first_idx = 0
            elif corners[0] == corners[4]:
                first_idx = 0 if corners[1] < corners[5] else 2
            else:
                first_idx = 2
        else:
            if corners[2] < corners[6]:
                first_idx = 1
            elif corners[2] == corners[6]:
                first_idx = 1 if corners[3] < corners[7] else 3
            else:
                first_idx = 3
        for j in range(4):
            idx = (first_idx + j) % 4
            sorted[i, j*2] = corners[idx*2]
            sorted[i, j*2+1] = corners[idx*2+1]
    return sorted


def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)


def is_image(filename):
    return any(filename.endswith(ext) for ext in [".bmp", ".png", ".jpg", ".jpeg", ".JPG"])


def rescale(im, target_size, max_size, keep_ratio, multiple=32):
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if keep_ratio:
        # method1
        im_scale = float(target_size) / float(im_size_min)  
        if np.round(im_scale * im_size_max) > max_size:     
            im_scale = float(max_size) / float(im_size_max)
        im_scale_x = np.floor(im.shape[1] * im_scale / multiple) * multiple / im.shape[1]
        im_scale_y = np.floor(im.shape[0] * im_scale / multiple) * multiple / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
        # method2
        # im_scale = float(target_size) / float(im_size_max)
        # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        # im_scale = np.array([im_scale, im_scale, im_scale, im_scale])

    else:
        target_size = int(np.floor(float(target_size) / multiple) * multiple)
        im_scale_x = float(target_size) / float(im_shape[1])
        im_scale_y = float(target_size) / float(im_shape[0])
        im = cv2.resize(im, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        im_scale = np.array([im_scale_x, im_scale_y, im_scale_x, im_scale_y])
    return im, im_scale


class Rescale(object):
    def __init__(self, target_size=600, max_size=2000, keep_ratio=True):
        self._target_size = target_size
        self._max_size = max_size
        self._keep_ratio = keep_ratio

    def __call__(self, im):
        if isinstance(self._target_size, list):
            random_scale_inds = npr.randint(0, high=len(self._target_size))
            target_size = self._target_size[random_scale_inds]
        else:
            target_size = self._target_size
        im, im_scales = rescale(im, target_size, self._max_size, self._keep_ratio)
        return im, im_scales


class Normailize(object):
    def __init__(self):
        # RGB: https://github.com/pytorch/vision/issues/223
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 均值和方差
        ])

    def __call__(self, im):
        im = self._transform(im)
        return im


class Reshape(object):
    def __init__(self, unsqueeze=True):
        self._unsqueeze = unsqueeze
        return

    def __call__(self, ims):
        if not torch.is_tensor(ims):
            ims = torch.from_numpy(ims.transpose((2, 0, 1)))
        if self._unsqueeze:
            ims = ims.unsqueeze(0)
        return ims

    
    
 

###
def show_dota_results(img_path,label_path):
    save_path = 'dota_res'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    merged_files = os.listdir(save_path)
    func = get_DOTA_points
    # for folder
    if os.path.isdir(img_path) and os.path.isdir(label_path):
        img_files = os.listdir(img_path)
        xml_files = os.listdir(label_path)
        img_files.sort()	
        xml_files.sort()
        
        img_names = [os.path.splitext(x)[0] for x in img_files]
        xml_names = [os.path.splitext(x)[0] for x in xml_files]
        for img_name in img_names:
            if img_name not in xml_names:
                img_files.remove(img_name+'.png')
#         import ipdb;ipdb.set_trace()
        assert len(img_files) == len(xml_files), 'Not matched between imgs and res!'
        iterations = zip(img_files,xml_files)
        for iter in iterations:
            if iter[0] in merged_files:
                continue
            assert os.path.splitext(iter[0])[0]==os.path.splitext(iter[1])[0],'unmatched images and labels!'   
            # object_coors = get_yolo_points(os.path.join(label_path,iter[1]), rotate=True)
            if not iter[0].endswith('.txt'):
                object_coors = func(os.path.join(label_path,iter[1]),True)
                if len(object_coors):
                    drawbox(os.path.join(img_path,iter[0]),object_coors, save_path =save_path )
                else:
                    print('No obj!')
    
    # for single img
    elif os.path.isfile(label_path):
        object_coors = func(os.path.join(label_path),rotate=False)
        if len(object_coors):
            drawbox(img_path,object_coors,False)
    else:
        print('Path Not Matched!!!')


def drawbox(img_path,object_coors,save_flag=True,save_path=None):
    print(img_path)

    img=cv2.imread(img_path,1)
    for coor in object_coors:
        img = cv2.polylines(img,[coor],True,(0,0,255),2)	
        if save_flag:
            cv2.imwrite(os.path.join(save_path,os.path.split(img_path)[1]), img)
        else: 
            cv2.imshow(img_path,img)
            cv2.moveWindow(img_path,100,100)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



def get_DOTA_points(label_path, rotate=False):
    if not os.path.exists(label_path):
        return []
    with open(label_path,'r') as f:        
        contents=f.read()
        lines=contents.split('\n')
        lines = [x for x in contents.split('\n')  if x]	 

        object_coors=[]	
        for object in lines:
            coors = object.split(' ')
            coors = [int(eval(x)) for x in coors[:-1]]
            x0 = coors[0]; y0 = coors[1]; x1 = coors[2]; y1 = coors[3]
            x2 = coors[4]; y2 = coors[5]; x3 = coors[6]; y3 = coors[7]
            object_coors.append(np.array([x0,y0,x1,y1,x2,y2,x3,y3]).reshape(4,2).astype(np.int32))
    return object_coors

def heading_from_quad(quad_pts):
    """
    根据 OBB 四点(4x2)计算长边方向航向(罗盘角，正北=0°、顺时针)和方向向量(像素坐标系).
    quad_pts: numpy array shape (4,2), 顺序无所谓（默认rbox_2_quad已按顺时针）
    返回: heading_deg(float in [0,360)), v_img(np.array([vx,vy]) 单位向量)
    """
    p = quad_pts.astype(np.float64)
    # 四条边向量
    e0 = p[1] - p[0]
    e1 = p[2] - p[1]
    e2 = p[3] - p[2]
    e3 = p[0] - p[3]
    L0, L1, L2, L3 = np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)

    # 选出“长边”的方向向量（e0/e2 与 e1/e3 比较）
    if (L0 + L2) >= (L1 + L3):
        v = e0 if L0 >= L2 else e2
    else:
        v = e1 if L1 >= L3 else e3

    # 归一化
    n = np.linalg.norm(v)
    if n < 1e-6:
        v = np.array([1.0, 0.0], dtype=np.float64)  # 退化保护：指向右
        n = 1.0
    v = v / n  # 图像坐标: x→右, y→下

    # 图坐标向量 -> 罗盘角(北=0°, 顺时针)
    # 对于图像坐标向量(vx, vy)，罗盘角 = atan2(vx, -vy)
    heading_deg = (np.degrees(np.arctan2(v[0], -v[1])) + 360.0) % 360.0
    return heading_deg, v

def draw_heading_arrow(img, heading_deg, anchor=None, length_ratio=0.12, color=(0,0,255), thickness=3):
    """
    根据罗盘角在图上画箭头；anchor 为起点(默认左上角偏内)，length_ratio 控制箭头长度。
    """
    H, W = img.shape[:2]
    L = int(max(10, min(H, W) * length_ratio))
    if anchor is None:
        x0, y0 = int(40), int(80)
    else:
        x0, y0 = int(anchor[0]), int(anchor[1])

    b_rad = np.degrees(0)  # not used; see below
    b_rad = np.deg2rad(heading_deg)
    dx = np.sin(b_rad)    # x向右为正
    dy = -np.cos(b_rad)   # y向下为正 -> 北向上取负号

    x1 = int(round(x0 + L * dx))
    y1 = int(round(y0 + L * dy))
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.25)

def _ensure_radians(theta):
    """把角度/弧度统一成弧度"""
    if abs(theta) > 2*np.pi:   # 很可能是度
        return np.deg2rad(theta)
    return theta

def estimate_wavelength_from_mask(bin_mask, heading_rad, min_area=20):
    """
    根据分割到的横波细线(bin_mask=0/1)与航向角(弧度)，估计横波波长（像素）。
    思路：连通域 -> 质心 -> 按航向投影 -> 相邻间距 -> 去异常 -> 中位数
    返回：dict{ spacings_px(list), lambda_px(float or None), n_crests(int) }
    """
    H, W = bin_mask.shape[:2]
    # 连通域（把很小的噪声滤掉）
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    centroids = centroids[1:] if num_cc > 1 else np.empty((0,2), dtype=np.float32)
    areas = stats[1:, cv2.CC_STAT_AREA] if num_cc > 1 else np.empty((0,), dtype=np.int32)
    keep = areas >= min_area
    centroids = centroids[keep]

    if len(centroids) < 2:
        return {"spacings_px": [], "lambda_px": None, "n_crests": int(len(centroids))}

    # 航向单位向量（图像坐标系：x向右，y向下）
    v = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float64)

    # 投影坐标（只用相对差值，因此绝对零点无关紧要）
    s = centroids @ v  # 形状 (N,)
    s_sorted = np.sort(s)
    spacings = np.diff(s_sorted)  # 相邻间距（像素）

    # 去除异常：用IQR或分位数裁剪
    if len(spacings) >= 3:
        q1, q3 = np.percentile(spacings, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        spacings = spacings[(spacings >= max(0, lo)) & (spacings <= hi)]

    if len(spacings) == 0:
        return {"spacings_px": [], "lambda_px": None, "n_crests": int(len(centroids))}

    lam_px = float(np.median(spacings))
    return {"spacings_px": spacings.tolist(), "lambda_px": lam_px, "n_crests": int(len(centroids))}


def draw_heading_arrow(img, heading_deg, anchor=None, length_ratio=0.12, color=(0, 0, 255), thickness=3):
    """
    根据罗盘角在图上画箭头；anchor 为起点(默认左上角偏内)，length_ratio 控制箭头长度。
    """
    H, W = img.shape[:2]
    L = int(max(10, min(H, W) * length_ratio))
    if anchor is None:
        x0, y0 = int(40), int(80)
    else:
        x0, y0 = int(anchor[0]), int(anchor[1])

    b_rad = np.deg2rad(heading_deg)
    dx = np.sin(b_rad)    # x向右为正
    dy = -np.cos(b_rad)   # y向下为正 -> 北向上取负号

    x1 = int(round(x0 + L * dx))
    y1 = int(round(y0 + L * dy))
    cv2.arrowedLine(img, (x0, y0), (x1, y1), color, thickness, tipLength=0.25)

def compass_wrap(deg):
    d = deg % 360.0
    if d < 0:
        d += 360.0
    if abs(d) < 1e-9 or abs(d - 360.0) < 1e-9:
        d = 0.0
    return d

def vec_to_compass_deg(vx, vy):
    # 图像坐标：x→右，y→下；罗盘角：北=0°，顺时针
    return compass_wrap(np.degrees(np.arctan2(vx, -vy)))

def _pick_cc_near_center(bin_mask, center, min_area=30, search_radius=99999):
    """
    从分割图里挑一块最像“尾迹”的连通域：面积≥min_area 且质心距离船中心最近（可加半径限制）。
    返回：mask_i(bool HxW)、质心(cx_i, cy_i)；若无则(None, None)
    """
    H, W = bin_mask.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    num, labels, stats, cents = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    best_id, best_d2 = -1, 1e18
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx_i, cy_i = float(cents[i, 0]), float(cents[i, 1])
        d2 = (cx_i - cx) ** 2 + (cy_i - cy) ** 2
        if d2 < best_d2 and np.sqrt(d2) <= search_radius:
            best_d2 = d2
            best_id = i
    if best_id < 0:
        return None, None
    cc_mask = (labels == best_id)
    cx_i, cy_i = float(cents[best_id, 0]), float(cents[best_id, 1])
    return cc_mask, (cx_i, cy_i)

def _heading_by_cc_normal(cc_mask, ship_center, require_corridor=False):
    """
    核心：用该连通域内“首尾两点”的连线求主轴向量 t，然后取法向 n。
    方向判定：令 n 指向船中心（从主轴中点指向船中心），得到唯一航向。
    返回：heading_deg(罗盘角), (vx,vy) 图像坐标下的单位向量
    """
    ys, xs = np.where(cc_mask)
    if xs.size < 2:
        return None, None

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)  # (N,2)
    # 先用 PCA 求主轴方向 t（比直接找最远点更稳），再用投影拿首尾点
    mean = pts.mean(axis=0)
    X = pts - mean
    C = (X.T @ X) / max(len(X) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    t = eigvecs[:, np.argmax(eigvals)]       # 主轴方向（二维向量）
    t = t / (np.linalg.norm(t) + 1e-9)
    # 投影，找首尾
    proj = X @ t
    i_min = int(np.argmin(proj))
    i_max = int(np.argmax(proj))
    p_min = pts[i_min]     # (x,y)
    p_max = pts[i_max]
    # “首尾连线”的方向向量（单位化）
    seg = p_max - p_min
    if np.linalg.norm(seg) < 1e-6:
        seg = t.copy()
    seg = seg / (np.linalg.norm(seg) + 1e-9)   # 记为主轴方向 t'（等价于 t 或反向）
    # 法向（两种之一）：n1 = 旋转90°，n2 = -n1
    n1 = np.array([ seg[1], -seg[0] ], dtype=np.float32)
    n1 = n1 / (np.linalg.norm(n1) + 1e-9)
    n2 = -n1

    # 以“首尾中点”为参考点，指向船中心，选择更接近船中心方向的法向作为“前进方向”
    mid = 0.5 * (p_min + p_max)           # (x,y)
    v_to_ship = np.array([ship_center[0] - mid[0], ship_center[1] - mid[1]], dtype=np.float32)
    # 谁与 v_to_ship 的夹角更小（点积更大），谁就是正向
    use_n = n1 if (np.dot(n1, v_to_ship) >= np.dot(n2, v_to_ship)) else n2

    heading_deg = vec_to_compass_deg(use_n[0], use_n[1])
    return float(heading_deg), (float(use_n[0]), float(use_n[1]))

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v) + 1e-9
    return v / n

def _long_edge_unit_from_quad(quad):
    """从 OBB 四点取长边方向（图像坐标；无向）"""
    q = quad.astype(np.float64)
    e0 = q[1] - q[0]; e1 = q[2] - q[1]; e2 = q[3] - q[2]; e3 = q[0] - q[3]
    lens = [np.linalg.norm(e0), np.linalg.norm(e1), np.linalg.norm(e2), np.linalg.norm(e3)]
    dirs = [e0, e1, e2, e3]
    return _unit(dirs[int(np.argmax(lens))])

def _vec_to_compass_deg(vx, vy):
    # 图像坐标：x→右, y→下；罗盘角：北=0° 顺时针
    d = (np.degrees(np.arctan2(vx, -vy)) + 360.0) % 360.0
    if abs(d) < 1e-9 or abs(d-360.0) < 1e-9: d = 0.0
    return d

def resolve_heading_by_midline_majority(
    bin_mask, obb_center, quad_pts,
    corridor_px=None, exclude_radius_px=None,
    min_fg_px=30, tau=0.05, debug=False
):
    """
    以 OBB 长边平行的“中线”（过 obb_center）为分界线，在走廊内统计两侧像素数，
    方向=从“多”指向“少”，并与长边平行。
    - corridor_px: 走廊半宽（垂直于长边方向的带宽）；None=按 OBB 短边 40% 自适应
    - exclude_radius_px: 排除中心邻域（避免船体干扰）；None=取 corridor_px 的 60%
    - min_fg_px: 有效前景像素的最小数量（不足则返回 None）
    - tau: 相对差阈值，|pos-neg|/(pos+neg) < tau 时认为不显著（不改变方向）
    返回: (heading_deg, v_img) 或 (None, None)
    """
    H, W = bin_mask.shape[:2]
    cx, cy = float(obb_center[0]), float(obb_center[1])

    # 长边方向 u 与法向 n
    u = _long_edge_unit_from_quad(quad_pts)
    n = np.array([u[1], -u[0]], dtype=np.float64)

    # 自适应走廊宽：取 OBB 短边的 40%
    if corridor_px is None:
        q = quad_pts.astype(np.float64)
        edges = [q[1]-q[0], q[2]-q[1], q[3]-q[2], q[0]-q[3]]
        short_edge = sorted([np.linalg.norm(e) for e in edges])[0]
        corridor_px = max(6, int(0.4 * short_edge))

    # 排除中心半径：默认 60% 的走廊宽
    if exclude_radius_px is None:
        exclude_radius_px = max(4, int(0.6 * corridor_px))

    ys, xs = np.where(bin_mask.astype(np.uint8) > 0)
    if xs.size < min_fg_px:
        if debug: print("[midline] too few fg pixels")
        return None, None

    xs = xs.astype(np.float64); ys = ys.astype(np.float64)
    dx = xs - cx
    dy = ys - cy

    # 轴向投影 s（沿长边方向）：中线两侧用 sign(s) 划分
    s = dx * u[0] + dy * u[1]
    # 到中线的垂距（走廊筛选）
    d_perp = np.abs(dx * n[0] + dy * n[1])

    in_cor = (d_perp <= corridor_px) & ((dx*dx + dy*dy) >= (exclude_radius_px**2))
    if not np.any(in_cor):
        if debug: print("[midline] no pixels in corridor")
        return None, None

    s_sel = s[in_cor]
    # 两侧像素计数
    pos = int(np.count_nonzero(s_sel >= 0))  # “前侧”
    neg = int(np.count_nonzero(s_sel <  0))  # “后侧”
    total = max(pos + neg, 1)
    diff = abs(pos - neg) / total

    if debug:
        print(f"[midline] pos={pos} neg={neg} diff={diff:.3f} corr={corridor_px}px excl={exclude_radius_px}px")

    # 不显著：不改变方向（返回 None, None 让上层回退）
    if diff < tau:
        return None, None

    # 方向=从“多”指向“少”，并与长边平行
    # 若前侧更多（pos>neg），应指向后侧（-u）；反之指向前侧（+u）
    v = -u if pos > neg else u

    heading_deg = _vec_to_compass_deg(v[0], v[1])
    return float(heading_deg), np.array([float(v[0]), float(v[1])], dtype=np.float32)

def resolve_heading_by_first_last(bin_mask, obb_center, quad_pts,
                                  min_area=20, debug=False):
    """
    逻辑：
      1) 取 OBB 长边无向单位向量 u，计算每个连通域质心相对船中心的投影 s = (c_cc - c_ship)·u
      2) 把连通域分成两侧：s>0 一侧、s<0 一侧；选择“总像素更多”的那一侧作为尾迹侧
      3) 在尾迹侧，按 |s| 从小到大排序：第 1 条 = 距船最近；最后 1 条 = 最远
      4) 若 第一条面积 >= 最后一条面积：航向取“从多到少”，即沿“从船→尾迹侧”的方向 (v = sign*u)
         若 第一条面积 <  最后一条面积：航向仍取“从多到少”，但此时从远→近，方向翻转 (v = -sign*u)

    返回： heading_deg(float), v_img(np.array([vx,vy]))；若无有效CC返回(None, None)
    """
    H, W = bin_mask.shape[:2]
    num, labels, stats, cents = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8
    )

    # 收集有效 CC（去掉背景与小噪声）
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx, cy = float(cents[i,0]), float(cents[i,1])
        comps.append({"id": i, "area": area, "cent": np.array([cx, cy], dtype=np.float64)})

    if len(comps) == 0:
        if debug: print("[first-last] no valid CC")
        return None, None

    # OBB 长边无向单位向量 u
    u = _long_edge_unit_from_quad(quad_pts)
    c_ship = np.array([float(obb_center[0]), float(obb_center[1])], dtype=np.float64)

    # 计算投影 s，并按两侧分组
    side_pos, side_neg = [], []
    for c in comps:
        d = c["cent"] - c_ship
        s = float(d[0]*u[0] + d[1]*u[1])
        item = {"area": c["area"], "s": s, "abs_s": abs(s)}
        if s >= 0:
            side_pos.append(item)
        else:
            side_neg.append(item)

    # 若两侧都没有，返回
    if len(side_pos) == 0 and len(side_neg) == 0:
        if debug: print("[first-last] no CC on either side after filtering")
        return None, None

    # 选择“总像素更多”的一侧作为尾迹侧
    sum_pos = sum(x["area"] for x in side_pos)
    sum_neg = sum(x["area"] for x in side_neg)
    if sum_pos >= sum_neg:
        wake_side = side_pos
        sign = +1.0   # v = +u 表示指向 s>0
        side_name = "pos(+)"
    else:
        wake_side = side_neg
        sign = -1.0  # v = -u 表示指向 s<0
        side_name = "neg(-)"

    if len(wake_side) < 1:
        if debug: print("[first-last] chosen side empty")
        return None, None

    # 按 |s| 从小到大（从近到远）
    wake_side.sort(key=lambda x: x["abs_s"])

    first_area = wake_side[0]["area"]
    last_area  = wake_side[-1]["area"]

    # 规则：始终“从多到少”
    # - 若 first >= last：从近(多) → 远(少)，方向 = 指向该侧（sign * u）
    # - 若 first <  last：从远(多) → 近(少)，方向 = 反向（-sign * u）
    if first_area >= last_area:
        v = sign * u
        reason = "first>=last → near→far"
    else:
        v = -sign * u
        reason = "first<last → far→near"

    heading_deg = vec_to_compass_deg(v[0], v[1])

    if debug:
        print(f"[first-last] side={side_name}, sum_pos={sum_pos}, sum_neg={sum_neg}")
        print(f"            first_area={first_area}, last_area={last_area} → {reason}")
        print(f"            heading={heading_deg:.2f}°")

    return float(heading_deg), np.array([float(v[0]), float(v[1])], dtype=np.float32)

