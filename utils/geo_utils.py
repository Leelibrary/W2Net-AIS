from pyproj import Transformer, Geod
import numpy as np
import rasterio

def pixel_segment_length_m(
    tif_path: str,
    col1: float,
    row1: float,
    col2: float,
    row2: float,
) -> float:
    """
    给定同一张 GeoTIFF 上的两个像素坐标 (col1,row1) 和 (col2,row2)，
    计算它们之间在 WGS84 椭球上的真实距离（单位：米）。
    """
    geod = Geod(ellps="WGS84")

    with rasterio.open(tif_path) as ds:
        if ds.crs is None or ds.transform is None:
            raise ValueError("该 TIFF 缺少 CRS 或仿射变换信息。")

        # 像素 -> 影像坐标系 (x,y)
        x1, y1 = ds.transform * (col1, row1)
        x2, y2 = ds.transform * (col2, row2)

        # 影像坐标系 -> WGS84(lon,lat)
        transformer = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lon1, lat1 = transformer.transform(x1, y1)
        lon2, lat2 = transformer.transform(x2, y2)

    # 计算测地线距离（米）
    az12, az21, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m