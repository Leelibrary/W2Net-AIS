import numpy as np
from typing import Dict, Tuple, List
import cv2
import rasterio
from pyproj import Geod, Transformer
from rasterio.transform import xy
import math

def haversine(lon1, lat1, lon2, lat2):
    """
    输入经纬度（度），返回两点大圆距离（米）
    """
    R = 6371000.0  # 地球平均半径（米）

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    d = R * c
    return d  # 米


def pixel_to_lonlat(tif_path: str, col: float, row: float) -> Tuple[float, float]:
    """
    像素坐标(col,row) -> WGS84(lon,lat)
    写法完全对齐你 compute_mask_geo_extent 里的 rc_to_lonlat。
    """
    with rasterio.open(tif_path) as ds:
        if ds.crs is None or ds.transform is None:
            raise ValueError("TIFF 缺少 CRS 或 transform 信息，无法做经纬度转换")

        transform = ds.transform
        crs = ds.crs
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        # 注意：这里输入的是 (col, row)
        x, y = transform * (col, row)
        lon, lat = transformer.transform(x, y)
        return float(lon), float(lat)


def geo_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    计算两点地理距离(米)，WGS84 椭球
    """
    geod = Geod(ellps='WGS84')
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return abs(dist)


def calculate_geo_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    计算两点地理距离(米)，WGS84 椭球.
    """
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return abs(distance)


def bresenham_line(x0: float, y0: float, x1: float, y1: float):
    """
    Bresenham 画线算法，返回从 (x0,y0) 到 (x1,y1) 的像素坐标序列.
    这里 x=col(列), y=row(行).
    """
    x0 = int(round(x0)); y0 = int(round(y0))
    x1 = int(round(x1)); y1 = int(round(y1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    xs, ys = [], []
    x, y = x0, y0

    while True:
        xs.append(x)
        ys.append(y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)

def pixel_to_xy(col, row, transform):
    """
    像素坐标 (col, row) -> 当前CRS下的平面坐标 (x, y)
    对你的 UTM Zone 10N 来说，单位就是米。
    """
    x, y = xy(transform, row, col)  # 注意顺序 row, col
    return float(x), float(y)


def pixel_distance_m(pt1, pt2, transform):
    """
    直接在 TIFF 的投影坐标系下算两点距离（米）：
    pt1, pt2: (x_pix, y_pix) 像素坐标（可以是 float）
    """
    c1 = int(round(pt1[0]))
    r1 = int(round(pt1[1]))
    c2 = int(round(pt2[0]))
    r2 = int(round(pt2[1]))

    x1, y1 = pixel_to_xy(c1, r1, transform)
    x2, y2 = pixel_to_xy(c2, r2, transform)

    dx = x2 - x1
    dy = y2 - y1
    return float((dx ** 2 + dy ** 2) ** 0.5)

def estimate_lambda_geo_from_mask(
    bin_mask: np.ndarray,
    v_img: np.ndarray,
    center: Tuple[float, float],
    tif_path: str,
    min_run_px: float = 3.0,   # 一个波峰在一维切线上至少要占多少像素才算有效
    min_pairs: int = 2         # 至少要有多少个“波峰间距”才能算出 λ
) -> Dict[str, object]:
    """
    按“逐个波峰/波谷 + 经纬度真实距离”估计 Kelvin 横波波长。

    步骤：
    1) 在船 OBB 中心处，沿“横波方向”的法线方向做一条一维剖面线；
       - 这里默认横波条纹大致垂直于航迹方向 v_img
       - 所以在图像里选 n = 旋转 90°(v_img) 作为剖面方向
    2) 在这条线上采样 bin_mask -> 得到一维 0/1 序列
    3) 找到这一维序列中连续的 1 段，每一段的几何中心视为一个“波峰中心”
    4) 对每一对相邻波峰中心：
         - 用像素坐标调用 pixel_segment_length_m(…, col1,row1,col2,row2)
         - 得到该段 λ 的真实距离（米）
       同时也用像素距离算出该段的 λ_px
    5) 对所有段的 λ_px 和 λ_m 求平均

    返回：
        {
            "lambda_px":  λ 的像素平均值 (float 或 None),
            "lambda_m":   λ 的米平均值 (float 或 None),
            "n_crests":   识别到的波峰个数 (int)
        }
    """
    H, W = bin_mask.shape[:2]

    # 1) 计算“横波剖面”的方向 n：令 n ⟂ v_img
    vx, vy = float(v_img[0]), float(v_img[1])
    n = np.array([-vy, vx], dtype=np.float64)   # 逆时针旋转 90°
    n_norm = np.linalg.norm(n) + 1e-9
    n /= n_norm

    cx, cy = float(center[0]), float(center[1])

    # 2) 决定剖面的长度范围：从中心向两侧各走 half_len 像素
    half_len = min(H, W) * 0.6    # 经验值，基本能盖住尾迹横波
    step = 0.5                    # 沿剖面采样的步长（像素）

    # t 从 -half_len 到 +half_len
    t_vals = np.arange(-half_len, half_len + step, step, dtype=np.float32)

    line_vals: List[int] = []
    coords: List[Tuple[float, float]] = []

    for t in t_vals:
        x = cx + n[0] * t
        y = cy + n[1] * t
        if 0 <= x < W and 0 <= y < H:
            # 最近邻采样
            ix = int(round(x))
            iy = int(round(y))
            line_vals.append(int(bin_mask[iy, ix]))
            coords.append((x, y))
        else:
            line_vals.append(0)
            coords.append((np.nan, np.nan))

    # 3) 在一维序列 line_vals 中找连续为 1 的区间 -> 波峰
    crest_indices: List[float] = []
    inside = False
    start_idx = 0

    # 在末尾补 0，用于“收尾”
    for i, v in enumerate(line_vals + [0]):
        if v == 1 and not inside:
            inside = True
            start_idx = i
        elif (v == 0 or np.isnan(v)) and inside:
            inside = False
            end_idx = i - 1
            if end_idx > start_idx:
                run_len = (end_idx - start_idx + 1) * step
                if run_len >= min_run_px:
                    mid_idx = (start_idx + end_idx) / 2.0
                    crest_indices.append(mid_idx)

    # 4) 将 crest_indices 映射回 2D 像素坐标
    crest_points: List[Tuple[float, float]] = []
    for mid_idx in crest_indices:
        idx_int = int(round(mid_idx))
        if 0 <= idx_int < len(coords):
            x, y = coords[idx_int]
            if not (np.isnan(x) or np.isnan(y)):
                crest_points.append((x, y))

    n_crests = len(crest_points)
    if n_crests < 2:
        return {"lambda_px": None, "lambda_m": None, "n_crests": int(n_crests)}

    # 5) 所有相邻波峰对 -> 像素 λ + 经纬度 λ
    lam_px_list: List[float] = []
    lam_m_list: List[float] = []

    from utils.geo_utils import pixel_segment_length_m  # ← 这里改成你实际的工具文件名

    for i in range(n_crests - 1):
        (x1, y1) = crest_points[i]
        (x2, y2) = crest_points[i + 1]

        # 像素距离
        dpx = float(np.hypot(x2 - x1, y2 - y1))
        if dpx <= 0:
            continue

        # 经纬度真实距离
        try:
            d_m = pixel_segment_length_m(
                tif_path,
                col1=x1, row1=y1,
                col2=x2, row2=y2
            )
        except Exception as e:
            print(f"[WARN] crest pair {i} λ(像素→经纬度) 转换失败: {e}")
            continue

        if d_m is None or d_m <= 0:
            continue

        lam_px_list.append(dpx)
        lam_m_list.append(float(d_m))

    if len(lam_px_list) == 0:
        return {"lambda_px": None, "lambda_m": None, "n_crests": int(n_crests)}

    lam_px_mean = float(np.mean(lam_px_list))
    lam_m_mean  = float(np.mean(lam_m_list))

    print(f"[INFO] crest-based λ_px_mean={lam_px_mean:.3f} px, λ_m_mean={lam_m_mean:.3f} m, pairs={len(lam_px_list)}")

    return {
        "lambda_px": lam_px_mean,
        "lambda_m": lam_m_mean,
        "n_crests": int(n_crests)
    }


def get_wave_direction_from_obb(quad_pts):
    """
    从 OBB 四点中得到“横波传播方向”(宽边中点连线方向)的单位向量
    即：长边方向 v_long 已知后，取与其垂直的短边方向 v_wave
    """
    q = quad_pts.astype(np.float64)
    edges = [q[1] - q[0], q[2] - q[1], q[3] - q[2], q[0] - q[3]]
    lens = [np.linalg.norm(e) for e in edges]
    long_idx = int(np.argmax(lens))  # 长边索引

    # 与长边相邻的一条边就是短边方向之一
    short_edge = edges[(long_idx + 1) % 4].astype(np.float64)
    v = short_edge / (np.linalg.norm(short_edge) + 1e-9)
    return v


def find_centerline_edge_point(mask_region, v_wave, center_pt, band_px=3):
    """
    在“经过 OBB 中心且方向为 v_wave 的直线”附近 (±band_px) 上，
    找到该连通域的“离船最远”的那个像素点，作为 crest 的离开边缘点。
    """
    ys, xs = np.where(mask_region > 0)
    if len(xs) == 0:
        return None

    pts = np.column_stack([xs, ys]).astype(np.float64)
    ref = np.array(center_pt, dtype=np.float64)
    rel = pts - ref  # 相对 OBB 中心的向量

    # 沿 v_wave 的投影（标量） -> 用来排序，表示“离船远近”
    proj = rel @ v_wave  # shape: (N,)

    # 到中心线的垂直距离：|v × rel|（2D 叉积的模）
    vx, vy = v_wave
    cross = vx * rel[:, 1] - vy * rel[:, 0]
    dist = np.abs(cross)

    # 只保留“靠近中心线”的点
    mask = dist <= band_px
    if np.any(mask):
        proj_sel = proj[mask]
        pts_sel = pts[mask]
    else:
        # 如果中心线附近没点，就退化为用所有点
        proj_sel = proj
        pts_sel = pts

    # 只考虑“船外”方向（proj > 0），如果没有就退化为全体
    valid = proj_sel > 0
    if np.any(valid):
        proj_valid = proj_sel[valid]
        pts_valid = pts_sel[valid]
        idx = np.argmax(proj_valid)  # 离船最远的一点
        edge_pt = pts_valid[idx]
    else:
        # 没有正投影点，直接取投影最大的
        idx = np.argmax(proj_sel)
        edge_pt = pts_sel[idx]

    return float(edge_pt[0]), float(edge_pt[1]), float(proj_sel[idx])

def extract_wave_crests_along_centerline(bin_mask, v_wave, obb_center,
                                         min_area=50, band_px=3, debug=False):
    """
    基于 OBB 中心 + v_wave（宽边中点连线方向）的中心线，
    提取每个连通域在这条线附近的“离船边缘点”，并按距离中心从近到远排序。
    """
    num_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bin_mask.astype(np.uint8), connectivity=8
    )

    wave_crests = []

    for i in range(1, num_cc):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        mask_i = (labels == i).astype(np.uint8)

        edge_info = find_centerline_edge_point(
            mask_region=mask_i,
            v_wave=v_wave,
            center_pt=obb_center,
            band_px=band_px
        )

        if edge_info is None:
            continue

        edge_x, edge_y, proj = edge_info

        if proj <= 0:
            # 在船“另一侧”的，就不算 Kelvin 尾迹这边了
            continue

        cx, cy = float(centroids[i, 0]), float(centroids[i, 1])

        wave_crests.append({
            'id': i,
            'area': area,
            'center': (cx, cy),
            'edge': (edge_x, edge_y),  # 只有一个“离船边缘点”
            'projection': proj         # 相对中心的距离（像素）
        })

    # 按“沿 v_wave 的投影”从近到远排序
    wave_crests.sort(key=lambda x: x['projection'])

    if debug:
        print(f"[提取波峰-中心线] 找到 {len(wave_crests)} 个有效波峰 (min_area={min_area}px)")

    return wave_crests


def calculate_wavelength_along_centerline(
    bin_mask: np.ndarray,
    v_wave: np.ndarray,
    obb_center: Tuple[float, float],
    tif_path: str,
    min_area: int = 20,
    band_px: int = 30,
    debug: bool = False
) -> Dict:
    """
    沿预测框中线方向，计算横波之间的平均间隔（物理距离，米）

    思路：
      1) 以 obb_center 为中心，沿 v_wave / -v_wave 各走一大段，得到一条长线段。
      2) 用 Bresenham 在这条线段上取所有像素点 (col,row)。
      3) 看这些点在 bin_mask 中是 0 还是 1，1 = 在横波内，0 = 在背景。
      4) 把连续为 1 的段记为一条“横波条带”（crest segment）。
      5) 对每一对相邻横波段：
           - 取上一条横波的“最后一个 1”的位置 idx_exit
           - 取下一条横波的“第一个 1”的位置 idx_enter_next
           - idx_exit 和 idx_enter_next 之间那一段就是“出分割 → 到下一条入分割”之间的间隔
         用这两个点的经纬度距离作为这一段间隔的物理长度。
      6) 对所有间隔取平均 / 标准差。

    返回:
        {
          'gap_mean_m': float or None,          # 横波间平均间隔距离（米）-- 你重点关心的量
          'gap_std_m':  float or None,          # 横波间间隔的标准差（米）
          'gaps_m':     List[float],            # 每一对横波之间的间隔（米）
          'n_crests':   int,                    # 本条中线穿过的横波条数
          'crest_segments': List[(s,e)]         # 每条横波在中线上对应的像素索引范围
        }
    """
    H, W = bin_mask.shape[:2]

    # 1) v_wave 归一化
    v = np.asarray(v_wave, dtype=np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-6:
        if debug:
            print("[calculate_wavelength_along_centerline] v_wave 近似为零向量，无法计算。")
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': 0,
            'crest_segments': []
        }
    v = v / v_norm  # 单位向量

    # 2) 构造“足够长”的线段（覆盖整个横波区域）
    #    用 图像对角线长度 的 1.5 倍 为半长，基本能跨过所有横波
    diag = math.hypot(H, W)
    half_len = diag * 0.75  # 半长（像素）

    cx, cy = obb_center  # 注意：这里默认 obb_center = (x=col, y=row)
    # 端点（可能会落在图外，后面再裁剪）
    x0 = cx - v[0] * half_len
    y0 = cy - v[1] * half_len
    x1 = cx + v[0] * half_len
    y1 = cy + v[1] * half_len

    # 3) 用 Bresenham 取出这条线上的所有像素
    xs, ys = bresenham_line(x0, y0, x1, y1)
    xs = np.asarray(xs, dtype=np.int32)
    ys = np.asarray(ys, dtype=np.int32)

    # 裁剪到图像内部
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    L = len(xs)
    if debug:
        print(f"[centerline] 采样点数: {L}")

    # 4) 取这条线上的 mask 值：1 = 横波内，0 = 背景
    vals = (bin_mask[ys, xs] > 0).astype(np.uint8)

    # 5) 找出每一条“横波段”：连续为 1 的 [start,end]
    crest_segments: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if vals[i] == 1:
            start = i
            while i + 1 < L and vals[i + 1] == 1:
                i += 1
            end = i
            # 可以按长度过滤掉太短的噪声段
            if (end - start + 1) >= min_area:
                crest_segments.append((start, end))
        i += 1

    n_crests = len(crest_segments)
    if debug:
        print(f"[centerline] 检测到横波条数: {n_crests}")

    # 少于2条横波，没法算“横波之间”的间隔
    if n_crests < 2:
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': n_crests,
            'crest_segments': crest_segments
        }

    # 6) 计算相邻横波之间的“间隔距离”
    gaps_m = []
    for k in range(n_crests - 1):
        start_k, end_k = crest_segments[k]
        start_next, end_next = crest_segments[k + 1]

        # 出前一条横波的位置：end_k（最后一个1）
        idx_exit = end_k
        # 进下一条横波的位置：start_next（第一个1）
        idx_enter = start_next

        if idx_enter <= idx_exit:
            # 理论上不会（中间总有一段0），防御性判断
            continue

        c1, r1 = xs[idx_exit], ys[idx_exit]
        c2, r2 = xs[idx_enter], ys[idx_enter]

        lon1, lat1 = pixel_to_lonlat(tif_path, c1, r1)
        lon2, lat2 = pixel_to_lonlat(tif_path, c2, r2)
        dist_m = calculate_geo_distance(lon1, lat1, lon2, lat2)
        gaps_m.append(dist_m)

        if debug:
            print(f"  gap {k}: 像素[{idx_exit}->{idx_enter}]  ≈ {dist_m:.3f} m")

    if len(gaps_m) == 0:
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': n_crests,
            'crest_segments': crest_segments
        }

    gap_mean_m = float(np.mean(gaps_m))
    gap_std_m = float(np.std(gaps_m))

    if debug:
        print(f"[centerline] 平均横波间隔: {gap_mean_m:.3f} ± {gap_std_m:.3f} m")

    return {
        'gap_mean_m': gap_mean_m,
        'gap_std_m': gap_std_m,
        'gaps_m': gaps_m,
        'n_crests': n_crests,
        'crest_segments': crest_segments
    }


def visualize_wave_edges_centerline(img, wave_crests, draw_connections=True):
    """沿中心线可视化每个 crest 的离船边缘点"""
    vis = img.copy()

    for idx, wc in enumerate(wave_crests):
        edge = wc['edge']
        center = wc['center']

        # 离船边缘点（红色）
        cv2.circle(vis, (int(edge[0]), int(edge[1])), 6, (0, 0, 255), -1)
        cv2.circle(vis, (int(edge[0]), int(edge[1])), 8, (255, 255, 255), 2)

        # 标注编号
        cv2.putText(vis, str(idx + 1), (int(center[0]) - 10, int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 连接相邻离船边缘点
    if draw_connections and len(wave_crests) > 1:
        for i in range(len(wave_crests) - 1):
            pt1 = wave_crests[i]['edge']
            pt2 = wave_crests[i + 1]['edge']
            cv2.line(vis, (int(pt1[0]), int(pt1[1])),
                     (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)

    return vis

def pixel_to_lonlat(tif_path, col, row):
    """像素坐标转经纬度"""
    with rasterio.open(tif_path) as src:
        lon, lat = src.xy(row, col)
    return lon, lat


def calculate_geo_distance(lon1, lat1, lon2, lat2):
    """计算两点地理距离(米)"""
    geod = Geod(ellps='WGS84')
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return abs(distance)

def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham 画线算法，返回从 (x0,y0) 到 (x1,y1) 的像素坐标序列
    这里 x=col(列), y=row(行)
    """
    x0 = int(round(x0)); y0 = int(round(y0))
    x1 = int(round(x1)); y1 = int(round(y1))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy

    xs, ys = [], []
    x, y = x0, y0

    while True:
        xs.append(x)
        ys.append(y)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(xs, dtype=np.int32), np.array(ys, dtype=np.int32)


def get_obb_direction_vector(quad_pts):
    """
    从 OBB 四点获取长边方向单位向量
    quad_pts: shape (4,2)，顺序为 rbox_2_quad 输出
    返回: (vx, vy)
    """
    q = np.asarray(quad_pts, dtype=np.float64)

    # 四条边向量
    edges = [
        q[1] - q[0],
        q[2] - q[1],
        q[3] - q[2],
        q[0] - q[3],
    ]

    lengths = [np.linalg.norm(e) for e in edges]

    # 最长边
    long_edge = edges[int(np.argmax(lengths))]

    v = long_edge / (np.linalg.norm(long_edge) + 1e-9)
    return v


def calculate_gap_along_obb_centerline(
    bin_mask: np.ndarray,
    quad: np.ndarray,
    obb_center: Tuple[float, float],
    tif_path: str,
    min_len_px: int = 5,
    debug: bool = False
) -> Dict:
    """
    用 OBB 的“长边方向旋转 90°”作为扫描方向，从 obb_center 出发
    做一条中心线，计算：

        入分割区域 -> 出分割区域 -> 再入下一条分割区域

    之间那一段的平均地理距离（横波之间的平均间隔）。

    参数:
        bin_mask   : 0/1 分割结果 (H,W)
        quad       : OBB 四点 (4,2)，来自 rbox_2_quad(...)
        obb_center : OBB 中心 (cx, cy)，注意顺序是 (x=col, y=row)
        tif_path   : 当前 tiff 路径，用于像素 -> 经纬度 -> 实际距离
        min_len_px : 一条横波在这条中心线上的最小长度(像素)，小于则视为噪声
        debug      : 是否打印详细信息

    返回:
        {
          'gap_mean_m': float or None,          # 横波之间平均间隔(米) —— 你重点关心
          'gap_std_m':  float or None,          # 间隔标准差(米)
          'gaps_m':     List[float],            # 每一对横波之间的间隔
          'n_crests':   int,                    # 中线穿过的横波条数
          'crest_segments': List[(s,e)],        # 每条横波在中心线上的像素索引范围
          'sample_xs':  np.ndarray,             # 中心线上采样的列坐标
          'sample_ys':  np.ndarray              # 中心线上采样的行坐标
        }
    """
    H, W = bin_mask.shape[:2]

    # === 1) 先拿到 OBB 长边方向 v_obb ===
    v_obb = get_obb_direction_vector(quad)   # (vx, vy)，已归一化

    # === 2) 旋转 90° 得到扫描方向 v_scan ===
    # 逆时针 90°: (-vy, vx)，如果方向反了，可以改成 (vy, -vx)
    v_scan = np.array([-v_obb[1], v_obb[0]], dtype=np.float64)
    v_norm = np.linalg.norm(v_scan)
    if v_norm < 1e-6:
        if debug:
            print("[centerline-obb] v_scan 近似为零向量，无法计算。")
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': 0,
            'crest_segments': [],
            'sample_xs': None,
            'sample_ys': None
        }
    v_scan /= v_norm

    # === 3) 以 OBB 中心为起点，沿 v_scan 构造一条足够长的线段 ===
    cx, cy = obb_center  # (x=col, y=row)
    # 用图像最大边的 0.8 倍作为半长，基本能覆盖整个尾迹
    half_len = max(H, W) * 0.8

    x0 = cx - v_scan[0] * half_len
    y0 = cy - v_scan[1] * half_len
    x1 = cx + v_scan[0] * half_len
    y1 = cy + v_scan[1] * half_len

    # === 4) Bresenham 采样中心线 ===
    xs, ys = bresenham_line(x0, y0, x1, y1)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)
    L = len(xs)

    if debug:
        print(f"[centerline-obb] 样本点数: {L}")

    # === 5) 读取这条线上对应的 mask 值：1=横波, 0=背景 ===
    vals = (bin_mask[ys, xs] > 0).astype(np.uint8)

    if debug:
        print(f"[short-midline] 这条线上的前景像素数: {int(vals.sum())}")



    crest_segments: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if vals[i] == 1:
            start = i
            while i + 1 < L and vals[i + 1] == 1:
                i += 1
            end = i
            # 不再按长度过滤，整个段都算一个横波
            crest_segments.append((start, end))
        i += 1

    n_crests = len(crest_segments)
    if debug:
        print(f"[centerline-obb] 检测到横波条数: {n_crests}")

    if n_crests < 2:
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': n_crests,
            'crest_segments': crest_segments,
            'sample_xs': xs,
            'sample_ys': ys
        }

    # === 7) 计算相邻横波之间的“间隔距离” ===
    gaps_m: List[float] = []
    for k in range(n_crests - 1):
        start_k, end_k = crest_segments[k]
        start_next, end_next = crest_segments[k + 1]

        # 出当前横波：最后一个1
        idx_exit = end_k
        # 进下一条横波：第一个1
        idx_enter = start_next

        if idx_enter <= idx_exit:
            continue

        c1, r1 = xs[idx_exit], ys[idx_exit]
        c2, r2 = xs[idx_enter], ys[idx_enter]

        lon1, lat1 = pixel_to_lonlat(tif_path, c1, r1)
        lon2, lat2 = pixel_to_lonlat(tif_path, c2, r2)
        dist_m = calculate_geo_distance(lon1, lat1, lon2, lat2)
        gaps_m.append(dist_m)

        if debug:
            print(f"  gap {k}: 像素[{idx_exit}->{idx_enter}]  ≈ {dist_m:.3f} m")

    if len(gaps_m) == 0:
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': n_crests,
            'crest_segments': crest_segments,
            'sample_xs': xs,
            'sample_ys': ys
        }

    gap_mean_m = float(np.mean(gaps_m))
    gap_std_m  = float(np.std(gaps_m))

    if debug:
        print(f"[centerline-obb] 平均横波间隔: {gap_mean_m:.3f} ± {gap_std_m:.3f} m")

    return {
        'gap_mean_m': gap_mean_m,
        'gap_std_m': gap_std_m,
        'gaps_m': gaps_m,
        'n_crests': n_crests,
        'crest_segments': crest_segments,
        'sample_xs': xs,
        'sample_ys': ys
    }



def get_obb_centerline_endpoints(quad: np.ndarray, extend_ratio: float = 0.5):
    """
    根据 OBB 四点，返回“宽边中点与中点连线”的两个端点（像素坐标，浮点数）.
    quad: shape (4, 2)，顺序是 rbox_2_quad 输出的四点顺序.
    extend_ratio: 沿该中线向两头再延长的比例（0.5 表示在原来的基础上各多延长 50%）

    返回:
        p1, p2: 两个端点 (x, y)，注意 x=col, y=row
    """
    q = quad.astype(np.float64)

    # 计算每一条边向量及长度
    edges = [q[1] - q[0], q[2] - q[1], q[3] - q[2], q[0] - q[3]]
    lens = [np.linalg.norm(e) for e in edges]

    # 找到最长的那条边的索引
    long_idx = int(np.argmax(lens))

    # 这条长边由点 i 和 i+1 组成，对边是 i+2 和 i+3
    i = long_idx
    j = (i + 1) % 4
    k = (i + 2) % 4
    l = (i + 3) % 4

    # 两条长边的中点
    mid1 = 0.5 * (q[i] + q[j])   # 第一条长边中点
    mid2 = 0.5 * (q[k] + q[l])   # 对边长边中点

    # 这是 OBB 的 "中心线"（与长边平行，穿过椭圆/船体中部）
    v = mid2 - mid1
    # 为了保险起见，让线段再向两头延长一点，避免刚好只在船体附近没穿到尾迹
    p1 = mid1 - extend_ratio * v
    p2 = mid2 + extend_ratio * v

    return p1, p2  # (x, y)


def get_short_edge_midline_endpoints(quad: np.ndarray,
                                     extend_ratio: float = 0.0) -> Tuple[Tuple[float, float],
                                                                          Tuple[float, float]]:
    """
    从 OBB 四点 quad (4,2) 中，找到两条“短边”，取它们的中点，
    用这两个中点连成一条线段（基础中线），再向两端各延长 extend_ratio 倍。

    extend_ratio=0 时，就是“短边的中点相连”这一条线。
    """
    q = np.asarray(quad, dtype=np.float64)  # (4,2)

    # 4 条边向量
    edges = [
        q[1] - q[0],  # 边0: 0->1
        q[2] - q[1],  # 边1: 1->2
        q[3] - q[2],  # 边2: 2->3
        q[0] - q[3],  # 边3: 3->0
    ]
    lens = [np.linalg.norm(e) for e in edges]

    # 最长边索引 → 对边索引
    long_idx = int(np.argmax(lens))
    long_opp = (long_idx + 2) % 4
    # 剩下两个就是短边
    short_idxs = sorted({0, 1, 2, 3} - {long_idx, long_opp})
    s0, s1 = short_idxs

    def edge_mid(idx):
        i0 = idx
        i1 = (idx + 1) % 4
        return 0.5 * (q[i0] + q[i1])

    mid0 = edge_mid(s0)
    mid1 = edge_mid(s1)

    v = mid1 - mid0
    p1 = mid0 - extend_ratio * v
    p2 = mid1 + extend_ratio * v

    return (float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))


def calc_wave_gaps_along_short_midline(
    bin_mask: np.ndarray,
    quad: np.ndarray,
    tif_path: str,
    debug: bool = False
) -> Dict:
    """
    模仿你对角线那段代码：
    只不过线不是“右下角→左上角”，
    而是 OBB 的“短边中点连线”。

    逻辑：
      1) 取 OBB 两条短边中点，连成一条线；
      2) 用 Bresenham 在这条线采样 (xs, ys)；
      3) vals = (bin_mask[ys, xs] > 0)  得到 0/1 序列；
      4) 每一段连续的 1 = 一条横波；
      5) 相邻横波之间的 0 段步数 = gap_steps；
      6) 用 GeoTIFF 的 transform 把整条线映射到投影坐标，算真实长度 line_len_m；
      7) m_per_step = line_len_m / (总步数)，gap_dists_m = gap_steps * m_per_step。
    """
    H, W = bin_mask.shape[:2]

    # 1) 得到“短边中点连线”的两个端点（像素坐标）
    (x0, y0), (x1, y1) = get_short_edge_midline_endpoints(quad, extend_ratio=0.0)

    # 2) 像素平面中的长度
    line_len_pix = math.hypot(x1 - x0, y1 - y0)

    # 3) Bresenham 采样
    xs, ys = bresenham_line(x0, y0, x1, y1)
    L = len(xs)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)

    if debug:
        print(f"[short-midline] 样本点数: {L}")
        print(f"[short-midline] 像素线长: {line_len_pix:.3f} px")

    # 4) 在线上取 mask 值：1 = 在横波里, 0 = 背景
    vals = (bin_mask[ys, xs] > 0).astype(np.uint8)
    if debug:
        print(f"[short-midline] 这条线上的前景像素数: {int(vals.sum())}")

    # 5) 找出每一段连续的 1（每一条被这条线穿过的横波）
    stripes: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if vals[i] == 1:
            start = i
            while i + 1 < L and vals[i + 1] == 1:
                i += 1
            end = i
            stripes.append((start, end))
        i += 1

    n_stripes = len(stripes)
    if debug:
        print(f"[short-midline] 检测到横波条数（沿这条线穿过的）: {n_stripes}")

    # 6) 计算相邻横波之间的“背景间隔步数”
    gap_steps: List[int] = []
    for k in range(len(stripes) - 1):
        end_k = stripes[k][1]
        start_next = stripes[k + 1][0]
        gap = max(0, start_next - end_k)
        if gap > 0:
            gap_steps.append(gap)

    if len(gap_steps) == 0:
        if debug:
            print("[short-midline] 横波之间没有有效间隔（可能只有一条横波或挨得很近）")
        return {
            'gap_mean_m': None,
            'gap_dists_m': [],
            'gap_steps': gap_steps,
            'n_stripes': n_stripes,
            'stripes': stripes,
            'xs': xs,
            'ys': ys,
            'line_len_pix': line_len_pix,
            'line_len_m': None,
        }

    # 7) 用 GeoTIFF 的 transform 算整条线的真实长度（单位：米）
    with rasterio.open(tif_path) as ds:
        transform = ds.transform
        crs = ds.crs

        # 像素 (col,row) -> 投影坐标 (X, Y)
        X0, Y0 = transform * (x0, y0)
        X1, Y1 = transform * (x1, y1)

        if crs is not None and crs.is_projected:
            # 投影坐标系：x,y 的单位就是“米”之类的线性单位
            line_len_m = math.hypot(X1 - X0, Y1 - Y0)
            if debug:
                print(f"[short-midline] CRS 为投影坐标系，线长(投影坐标): {line_len_m:.3f} m")
        else:
            # 地理坐标系（经纬度），退回到 haversine
            # 注意 haversine(lon, lat)；这里假定 X=lon, Y=lat
            line_len_m = haversine(X0, Y0, X1, Y1)
            if debug:
                print(f"[short-midline] CRS 为地理坐标系，线长(大圆距离): {line_len_m:.3f} m")

    # 8) 和你对角线代码一样：总步数 -> 每步多少米
    steps_total = max(1, L - 1)
    m_per_step = line_len_m / steps_total

    gap_dists_m = [g * m_per_step for g in gap_steps]
    n_gap = len(gap_dists_m)

    # ===== 去掉 1 个最小值 + 1 个最大值，再求平均 =====
    if n_gap >= 3:
        gap_sorted = np.sort(np.array(gap_dists_m, dtype=np.float64))
        gap_dists_used = gap_sorted[1:-1]  # 总数 = n_gap - 2
        gap_mean_m = float(np.sum(gap_dists_used) / (n_gap - 2))
    else:
        # gap 数量不足，无法去极值
        gap_dists_used = gap_dists_m
        gap_mean_m = float(np.mean(gap_dists_used)) if n_gap > 0 else None

    if debug:
        print("\n=== 相邻横波之间的间隔（沿短边中点连线方向） ===")
        for idx, (steps, dist_m) in enumerate(zip(gap_steps, gap_dists_m)):
            print(f"  gap {idx}: {steps} 像素步 ≈ {dist_m:.3f} m")
        print(f"\n  横波间隔平均距离: {gap_mean_m:.3f} m")

    return {
        'gap_mean_m': gap_mean_m,
        'gap_dists_m': gap_dists_m,
        'gap_dists_used': gap_dists_used,
        'gap_steps': gap_steps,
        'n_stripes': n_stripes,
        'stripes': stripes,
        'xs': xs,
        'ys': ys,
        'line_len_pix': line_len_pix,
        'line_len_m': line_len_m,
    }

def remove_outliers_mad(values, thresh=3.0):
    """
    使用 MAD 方法剔除异常值
    values: list 或 array of 波距（单位：米）
    thresh: 3.0 是常用阈值，越小越严格

    返回：过滤后的 array
    """
    if len(values) < 3:
        return np.array(values)  # 点太少，不过滤

    values = np.array(values, dtype=float)
    median = np.median(values)
    abs_dev = np.abs(values - median)
    mad = np.median(abs_dev)

    if mad == 0:
        return values  # 全部一样，无异常

    z = abs_dev / (mad * 1.4826)  # 1.4826 是常用尺度因子，使其逼近标准差
    return values[z < thresh]

def calculate_gap_along_short_midline(
    bin_mask: np.ndarray,
    quad: np.ndarray,
    tif_path: str,
    min_len_px: int = 5,
    debug: bool = False
) -> Dict:
    """
    以 OBB 的“短边中点连线”作为扫描线，沿该线计算：
        入分割区域 -> 出分割区域 -> 再入下一条分割区域
    之间那一段的地理距离（横波之间的间隔），并给出平均值。

    参数:
        bin_mask   : 0/1 分割结果 (H,W)
        quad       : OBB 四点 (4,2)，来自 rbox_2_quad(bbox[:5]).reshape(4,2)
        tif_path   : 对应 GeoTIFF 路径
        min_len_px : 一条横波在这条线上的最小长度(像素)，小于则视为噪声
        debug      : 是否打印过程信息

    返回:
        {
          'gap_mean_m': float or None,          # 横波间平均间隔(米)
          'gap_std_m':  float or None,          # 间隔标准差
          'gaps_m':     List[float],            # 每一对横波间的间隔
          'n_crests':   int,                    # 检测到的横波条数
          'crest_segments': List[(s,e)],        # 每条横波对应的 [start,end] 像素索引
          'sample_xs':  np.ndarray,             # 扫描线上采样的列
          'sample_ys':  np.ndarray,             # 扫描线上采样的行
        }
    """
    H, W = bin_mask.shape[:2]

    # 1) 以“短边中点连线”作为扫描线，得到两个端点
    (x0, y0), (x1, y1) = get_short_edge_midline_endpoints(quad, extend_ratio=0.5)

    # 2) Bresenham 在像素格上采样
    xs, ys = bresenham_line(x0, y0, x1, y1)
    xs = np.clip(xs, 0, W - 1)
    ys = np.clip(ys, 0, H - 1)
    L = len(xs)

    if debug:
        print(f"[short-midline] 样本点数: {L}")

    # 3) 在这条线上查看 mask：1=在横波内，0=背景
    vals = (bin_mask[ys, xs] > 0).astype(np.uint8)

    # 4) 找连续的 1 段，视为“被扫描线切到的横波条带”
    crest_segments: List[Tuple[int, int]] = []
    i = 0
    while i < L:
        if vals[i] == 1:
            start = i
            while i + 1 < L and vals[i + 1] == 1:
                i += 1
            end = i
            if (end - start + 1) >= min_len_px:
                crest_segments.append((start, end))
        i += 1

    n_crests = len(crest_segments)
    if debug:
        print(f"[short-midline] 检测到横波条数: {n_crests}")

    # 横波少于 2 条，没法算“横波之间”的间隔
    if n_crests < 2:
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': n_crests,
            'crest_segments': crest_segments,
            'sample_xs': xs,
            'sample_ys': ys
        }

    # 5) 计算相邻横波之间的“间隔”：出当前横波 end_k -> 进下一条 start_next
    gaps_m: List[float] = []
    for k in range(n_crests - 1):
        start_k, end_k = crest_segments[k]
        start_next, end_next = crest_segments[k + 1]

        idx_exit  = end_k        # 出当前横波
        idx_enter = start_next   # 入下一条横波

        if idx_enter <= idx_exit:
            continue

        c1, r1 = xs[idx_exit], ys[idx_exit]
        c2, r2 = xs[idx_enter], ys[idx_enter]

        lon1, lat1 = pixel_to_lonlat(tif_path, c1, r1)
        lon2, lat2 = pixel_to_lonlat(tif_path, c2, r2)
        dist_m = calculate_geo_distance(lon1, lat1, lon2, lat2)
        gaps_m.append(dist_m)

        if debug:
            print(f"  gap {k}: 像素[{idx_exit}->{idx_enter}]  ≈ {dist_m:.3f} m")

    if len(gaps_m) == 0:
        return {
            'gap_mean_m': None,
            'gap_std_m': None,
            'gaps_m': [],
            'n_crests': n_crests,
            'crest_segments': crest_segments,
            'sample_xs': xs,
            'sample_ys': ys
        }

    gap_mean_m = float(np.mean(gaps_m))
    gap_std_m  = float(np.std(gaps_m))

    if debug:
        print(f"[short-midline] 平均横波间隔: {gap_mean_m:.3f} ± {gap_std_m:.3f} m")

    return {
        'gap_mean_m': gap_mean_m,
        'gap_std_m': gap_std_m,
        'gaps_m': gaps_m,
        'n_crests': n_crests,
        'crest_segments': crest_segments,
        'sample_xs': xs,
        'sample_ys': ys
    }

def is_image_file(fname):
    return fname.lower().endswith((
        '.tif', '.tiff',
        '.jpg', '.jpeg',
        '.png', '.bmp'
    ))

def hex_to_bgr(color_hex: str):
    """
    '#C5A2CE' -> (206, 162, 197)  # BGR
    """
    color_hex = color_hex.lstrip('#')
    if len(color_hex) != 6:
        raise ValueError(f'非法颜色格式: {color_hex}')
    r = int(color_hex[0:2], 16)
    g = int(color_hex[2:4], 16)
    b = int(color_hex[4:6], 16)
    return (b, g, r)
