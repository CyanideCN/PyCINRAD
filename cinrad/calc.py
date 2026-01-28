# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import datetime
import time
from typing import *
from functools import wraps

import numpy as np
from xarray import DataArray, Dataset

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import KDTree

from cinrad.utils import *
from cinrad.grid import grid_2d, resample
from cinrad.projection import height, get_coordinate
from cinrad.error import RadarCalculationError
from cinrad._typing import Volume_T
from cinrad.common import get_dtype
from cinrad.hca import hydro_class as _hca

__all__ = [
    "quick_cr",
    "quick_et",
    "quick_vil",
    "VCS",
    "quick_vild",
    "GridMapper",
    "hydro_class",
    "CAPPI",
]


def require(var_names: List[str]) -> Callable:
    def wrap(func: Callable) -> Callable:
        @wraps(func)
        def deco(*args, **kwargs) -> Any:
            if len(args) == 1:
                varset = args[0]
            else:
                varset = list(args)
            if isinstance(varset, Dataset):
                var_list = list(varset.keys())
            elif isinstance(varset, list):
                var_list = list()
                for var in varset:
                    var_list += list(var.keys())
                var_list = list(set(var_list))
            for v in var_names:
                if v not in var_list:
                    raise ValueError(
                        "Function {} requires variable {}".format(func.__name__, v)
                    )
            return func(*args, **kwargs)

        return deco

    return wrap


def _extract(r_list: Volume_T, dtype: str) -> tuple:
    if len(set(i.range for i in r_list)) > 1:
        raise ValueError("Input radials must have same data range")
    adim_shape = set(i.dims["azimuth"] for i in r_list)
    if max(adim_shape) > 400:
        # CC radar
        adim_interp_to = 512
    else:
        adim_interp_to = 360
    r_data = list()
    elev = list()
    for i in r_list:
        x, d, a = resample(
            i[dtype].values,
            i["distance"].values,
            i["azimuth"].values,
            i.tangential_reso,
            adim_interp_to,
        )
        r_data.append(x)
        elev.append(i.elevation)
    data = np.concatenate(r_data).reshape(
        len(r_list), r_data[0].shape[0], r_data[0].shape[1]
    )
    return data, d, a, np.array(elev)


@require(["REF"])
def quick_cr(r_list: Volume_T, resolution: tuple = (1000, 1000)) -> Dataset:
    r"""Calculate composite reflectivity

    Args:
        r_list (list(xarray.Dataset)): Reflectivity data.

    Returns:
        xarray.Dataset: composite reflectivity
    """
    r_data = list()
    # Get grid from the first tilt
    r, x, y = grid_2d(
        r_list[0]["REF"].values,
        r_list[0]["longitude"].values,
        r_list[0]["latitude"].values,
        resolution=resolution,
    )
    r_data.append(r)
    for i in r_list[1:]:
        r, x, y = grid_2d(
            i["REF"].values,
            i["longitude"].values,
            i["latitude"].values,
            x_out=x,
            y_out=y,
            resolution=resolution,
        )
        r_data.append(r)
    cr = np.nanmax(r_data, axis=0)
    ret = Dataset({"CR": DataArray(cr, coords=[y, x], dims=["latitude", "longitude"])})
    ret.attrs = i.attrs
    ret.attrs["elevation"] = 0
    return ret


@require(["REF"])
def quick_et(r_list: Volume_T) -> Dataset:
    r"""Calculate echo tops

    Args:
        r_list (list(xarray.Dataset)): Reflectivity data.

    Returns:
        xarray.Dataset: echo tops
    """
    r_data, d, a, elev = _extract(r_list, "REF")
    r_data[np.isnan(r_data)] = 0
    i = r_list[0]
    et = echo_top(
        r_data.astype(np.double), d.astype(np.double), elev.astype(np.double), 0.0
    )
    azimuth = a[:, 0]
    distance = d[0]
    ret = Dataset(
        {
            "ET": DataArray(
                np.ma.masked_less(et, 2),
                coords=[azimuth, distance],
                dims=["azimuth", "distance"],
            )
        }
    )
    ret.attrs = i.attrs
    ret.attrs["elevation"] = 0
    lon, lat = get_coordinate(distance, azimuth, 0, i.site_longitude, i.site_latitude)
    ret["longitude"] = (["azimuth", "distance"], lon)
    ret["latitude"] = (["azimuth", "distance"], lat)
    return ret


@require(["REF"])
def quick_vil(r_list: Volume_T) -> Dataset:
    r"""Calculate vertically integrated liquid.

    This algorithm process data in polar coordinates, which avoids the loss of
    data. By default, this function calls low-level function `vert_integrated_liquid`
    in C-extension. If the C-extension is not available, the python version will
    be used instead but with much slower speed.

    Args:
        r_list (list(xarray.Dataset)): Reflectivity data.

    Returns:
        xarray.Dataset: vertically integrated liquid
    """
    r_data, d, a, elev = _extract(r_list, "REF")
    r_data[np.isnan(r_data)] = 0
    i = r_list[0]
    vil = vert_integrated_liquid(
        r_data.astype(np.double), d.astype(np.double), elev.astype(np.double)
    )
    azimuth = a[:, 0]
    distance = d[0]
    ret = Dataset(
        {
            "VIL": DataArray(
                np.ma.masked_less(vil, 0.1),
                coords=[azimuth, distance],
                dims=["azimuth", "distance"],
            )
        }
    )
    ret.attrs = i.attrs
    ret.attrs["elevation"] = 0
    lon, lat = get_coordinate(distance, azimuth, 0, i.site_longitude, i.site_latitude)
    ret["longitude"] = (["azimuth", "distance"], lon)
    ret["latitude"] = (["azimuth", "distance"], lat)
    return ret


def quick_vild(r_list: Volume_T) -> Dataset:
    r"""Calculate vertically integrated liquid density.

    By default, this function calls low-level function `vert_integrated_liquid`
    in C-extension. If the C-extension is not available, the python version will
    be used instead but with much slower speed.

    Args:
        r_list (list(xarray.Dataset)): Reflectivity data.

    Returns:
        xarray.Dataset: Vertically integrated liquid
    """
    r_data, d, a, elev = _extract(r_list, "REF")
    i = r_list[0]
    vild = vert_integrated_liquid(
        r_data.astype(np.double),
        d.astype(np.double),
        elev.astype(np.double),
        density=True,
    )
    azimuth = a[:, 0]
    distance = d[0]
    ret = Dataset(
        {
            "VILD": DataArray(
                np.ma.masked_less(vild, 0.1),
                coords=[azimuth, distance],
                dims=["azimuth", "distance"],
            )
        }
    )
    ret.attrs = i.attrs
    ret.attrs["elevation"] = 0
    lon, lat = get_coordinate(distance, azimuth, 0, i.site_longitude, i.site_latitude)
    ret["longitude"] = (["azimuth", "distance"], lon)
    ret["latitude"] = (["azimuth", "distance"], lat)
    return ret


def polar_to_xy(field: Dataset, resolution: tuple = (1000, 1000)) -> Dataset:
    r"""
    Interpolate single volume data in polar coordinates into geographic coordinates

    将单仰角数据从极坐标插值转换为经纬度坐标

    Args:
        field (xarray.Dataset): Original radial.

    Returns:
        xarray.Dataset: Interpolated data in grid
    """
    dtype = get_dtype(field)
    r, x, y = grid_2d(
        field[dtype].values,
        field["longitude"].values,
        field["latitude"].values,
        resolution=resolution,
    )
    ret = Dataset({dtype: DataArray(r, coords=[y, x], dims=["latitude", "longitude"])})
    ret.attrs = field.attrs
    return ret


class VCS(object):
    r"""
    Class performing vertical cross-section calculation

    Args:
        r_list (list(xarray.Dataset)): The whole volume scan.
    """

    def __init__(self, r_list: Volume_T):
        el = [i.elevation for i in r_list]
        if len(el) != len(set(el)):
            self.rl = list()
            el_list = list()
            for data in r_list:
                if data.elevation not in el_list:
                    self.rl.append(data)
                    el_list.append(data.elevation)
        else:
            self.rl = r_list
        self.dtype = get_dtype(r_list[0])
        self.x, self.y, self.h, self.r = self._geocoor()
        self.attrs = r_list[0].attrs

    def _geocoor(self) -> Tuple[list]:
        r_data = list()
        x_data = list()
        y_data = list()
        h_data = list()
        for i in self.rl:
            _lon = i["longitude"].values
            _lat = i["latitude"].values
            r, x, y = grid_2d(i[self.dtype].values, _lon, _lat)
            r, x, y = grid_2d(i[self.dtype].values, _lon, _lat)
            r_data.append(r)
            x_data.append(x)
            y_data.append(y)
            hgh_grid, x, y = grid_2d(i["height"].values, _lon, _lat)
            h_data.append(hgh_grid)
        return x_data, y_data, h_data, r_data

    def _get_section(
        self, stp: Tuple[float, float], enp: Tuple[float, float], spacing: int
    ) -> Dataset:
        r_sec = list()
        h_sec = list()
        for x, y, h, r in zip(self.x, self.y, self.h, self.r):
            d_x = DataArray(r, [("lat", y), ("lon", x)])
            d_h = DataArray(h, [("lat", y), ("lon", x)])
            x_new = DataArray(np.linspace(stp[0], enp[0], spacing), dims="z")
            y_new = DataArray(np.linspace(stp[1], enp[1], spacing), dims="z")
            r_section = d_x.interp(lon=x_new, lat=y_new)
            h_section = d_h.interp(lon=x_new, lat=y_new)
            r_sec.append(r_section)
            h_sec.append(h_section)
        r = np.asarray(r_sec)
        h = np.asarray(h_sec)
        x = np.linspace(0, 1, spacing) * np.ones(r.shape[0])[:, np.newaxis]
        ret = Dataset(
            {
                self.dtype: DataArray(r, dims=["distance", "tilt"]),
                "y_cor": DataArray(h, dims=["distance", "tilt"]),
                "x_cor": DataArray(x, dims=["distance", "tilt"]),
            }
        )
        r_attr = self.attrs.copy()
        del r_attr["elevation"], r_attr["tangential_reso"], r_attr["range"]
        r_attr["start_lon"] = stp[0]
        r_attr["start_lat"] = stp[1]
        r_attr["end_lon"] = enp[0]
        r_attr["end_lat"] = enp[1]
        ret.attrs = r_attr
        return ret

    def get_section(
        self,
        start_polar: Optional[Tuple[float, float]] = None,
        end_polar: Optional[Tuple[float, float]] = None,
        start_cart: Optional[Tuple[float, float]] = None,
        end_cart: Optional[Tuple[float, float]] = None,
        spacing: int = 500,
    ) -> Dataset:
        r"""
        Get cross-section data from input points

        Args:
            start_polar (tuple): polar coordinates of start point i.e.(distance, azimuth)

            end_polar (tuple): polar coordinates of end point i.e.(distance, azimuth)

            start_cart (tuple): geographic coordinates of start point i.e.(longitude, latitude)

            end_cart (tuple): geographic coordinates of end point i.e.(longitude, latitude)

        Returns:
            xarray.Dataset: Cross-section data
        """
        if start_polar and end_polar:
            stlat = self.rl[0].site_latitude
            stlon = self.rl[0].site_longitude
            stp = np.round(
                get_coordinate(
                    start_polar[0], np.deg2rad(start_polar[1]), 0, stlon, stlat
                ),
                2,
            )
            enp = np.round(
                get_coordinate(end_polar[0], np.deg2rad(end_polar[1]), 0, stlon, stlat),
                2,
            )
        elif start_cart and end_cart:
            stp = start_cart
            enp = end_cart
        else:
            raise RadarCalculationError("Invalid input")
        return self._get_section(stp, enp, spacing)


class GridMapper(object):
    r"""
    This class can merge scans from different radars to a single cartesian grid.
    support BR or CR or any list(xarray.Dataset).
    merge_xy method inspiration comes from OLDLee_GIFT@bilibili.

    Args:
        fields (list(xarray.Dataset)): Lists of scans to be merged.

        max_dist (int, float): The maximum distance in kdtree searching.

    Example:
        >>> gm = GridMapper([r1, r2, r3])
        >>> grid = gm(0.1)
    """

    def __init__(self, fields: Volume_T, max_dist: Number_T = 0.1):
        # Process data type
        self.dtype = get_dtype(fields[0])
        # Process time
        t_arr = np.array(
            [
                time.mktime(
                    datetime.datetime.strptime(
                        i.scan_time, "%Y-%m-%d %H:%M:%S"
                    ).timetuple()
                )
                for i in fields
            ]
        )
        if (t_arr.max() - t_arr.min()) / 60 > 10:
            raise RadarCalculationError(
                "Time difference of input data should not exceed 10 minutes"
            )
        mean_time = t_arr.mean()
        mean_dtime = datetime.datetime(*time.localtime(int(mean_time))[:6])
        time_increment = 10
        time_rest = mean_dtime.minute % time_increment
        if time_rest > time_increment / 2:
            mean_dtime += datetime.timedelta(minutes=(time_increment - time_rest))
        else:
            mean_dtime -= datetime.timedelta(minutes=time_rest)
        self.scan_time = mean_dtime
        self.lon_ravel = np.hstack([i["longitude"].values.ravel() for i in fields])
        self.lat_ravel = np.hstack([i["latitude"].values.ravel() for i in fields])
        self.is_polar = "distance" in fields[0].coords
        if self.is_polar:
            self.data_ravel = np.ma.hstack(
                [i[self.dtype].values.ravel() for i in fields]
            )
            self.dist_ravel = np.hstack(
                [
                    np.broadcast_to(i["distance"], i["longitude"].shape).ravel()
                    for i in fields
                ]
            )
            self.tree = KDTree(np.dstack((self.lon_ravel, self.lat_ravel))[0])
        self.md = max_dist
        self.attr = fields[0].attrs.copy()
        self.fields = fields

    def _process_grid(self, x_step: Number_T, y_step: Number_T) -> Tuple[np.ndarray]:
        x_lower = np.round(self.lon_ravel.min(), 2)
        x_upper = np.round(self.lon_ravel.max(), 2)
        y_lower = np.round(self.lat_ravel.min(), 2)
        y_upper = np.round(self.lat_ravel.max(), 2)
        x_grid = np.arange(x_lower, x_upper + x_step, x_step)
        y_grid = np.arange(y_lower, y_upper + y_step, y_step)
        return np.meshgrid(x_grid, y_grid)

    def _map_points(self, x: np.ndarray, y: np.ndarray) -> np.ma.MaskedArray:
        _MAX_RETURN = 5
        _FILL_VALUE = -1e5
        xdim, ydim = x.shape
        _, idx = self.tree.query(
            np.dstack((x.ravel(), y.ravel()))[0],
            distance_upper_bound=self.md,
            k=_MAX_RETURN,
        )
        idx_all = idx.ravel()
        data_indexing = np.append(self.data_ravel, _FILL_VALUE)
        dist_indexing = np.append(self.dist_ravel, 0)
        target_rad = np.ma.masked_equal(data_indexing[idx_all], _FILL_VALUE)
        weight = dist_indexing[idx_all]
        inp = target_rad.reshape(xdim, ydim, _MAX_RETURN)
        wgt = weight.reshape(xdim, ydim, _MAX_RETURN)
        return np.ma.average(inp, weights=1 / wgt, axis=2)

    def _merge_xy(self, x: np.ndarray, y: np.ndarray) -> Dataset:
        # interpolate datas to full grid
        r_data = list()
        for field in self.fields:
            field_grid = field.interp(longitude=x[0], latitude=y[:, 0], method="linear")
            r_data.append(field_grid[self.dtype].values)
        # select max value in each grid
        r_data_max = np.nanmax(r_data, axis=0)
        ret = Dataset(
            {
                self.dtype: DataArray(
                    r_data_max, coords=[y[:, 0], x[0]], dims=["latitude", "longitude"]
                )
            }
        )
        # interpolate Nan values
        lat_interp = ret[self.dtype].interpolate_na(
            "latitude", method="linear", limit=len(self.fields)
        )
        lon_interp = lat_interp.interpolate_na(
            "longitude", method="nearest", limit=len(self.fields)
        )
        grid = lon_interp.interp(longitude=x[0], latitude=y[:, 0], method="linear")
        return grid

    def __call__(self, step: Number_T) -> Dataset:
        r"""
        Args:
            step (int, float): Output grid spacing.

        Returns:
            xarray.Dataset: Merged grid data.
        """
        x, y = self._process_grid(step, step)
        if self.is_polar:
            grid = self._map_points(x, y)

        else:
            grid = self._merge_xy(x, y)
        grid = np.ma.masked_outside(grid, 0.1, 100)
        ret = Dataset(
            {
                self.dtype: DataArray(
                    grid, coords=[y[:, 0], x[0]], dims=["latitude", "longitude"]
                )
            }
        )
        r_attr = self.attr
        # Keep this attribute temporarily waiting for future fix
        r_attr["tangential_reso"] = np.nan
        r_attr["elevation"] = 0
        r_attr["site_name"] = "RADMAP"
        r_attr["site_code"] = "RADMAP"
        r_attr["scan_time"] = self.scan_time.strftime("%Y-%m-%d %H:%M:%S")
        for k in ["site_longitude", "site_latitude", "nyquist_vel"]:
            if k in r_attr:
                del r_attr[k]
        ret.attrs = r_attr
        return ret


@require(["REF", "ZDR", "RHO", "KDP"])
def hydro_class(
    z: Dataset, zdr: Dataset, rho: Dataset, kdp: Dataset, band: str = "S"
) -> Dataset:
    r"""Hydrometeor classification

    Args:
        z (xarray.Dataset): Reflectivity data.

        zdr (xarray.Dataset): Differential reflectivity data.

        rho (xarray.Dataset): Cross-correlation coefficient data.

        kdp (xarray.Dataset): Specific differential phase data.

        band (str): Band of the radar, default to S.

    Returns:
        xarray.Dataset: Classification result.
    """
    z_data = z["REF"].values
    zdr_data = zdr["ZDR"].values
    rho_data = rho["RHO"].values
    kdp_data = kdp["KDP"].values
    result = _hca(
        z_data.ravel(), zdr_data.ravel(), rho_data.ravel(), kdp_data.ravel(), band=band
    )
    result = result.reshape(z_data.shape).astype(float)
    result[np.isnan(z_data)] = np.nan
    hcl = z.copy()
    hcl["cHCL"] = (["azimuth", "distance"], result)
    del hcl["REF"]
    return hcl


class CAPPI(object):
    r"""
    CAPPI (Constant Altitude Plan Position Indicator) 等高面计算类
    
    使用3D体素插值方法，复用库内现有的高度计算和坐标转换函数。

    Args:
        r_list: 各仰角的反射率数据列表
        verbose: 是否打印性能信息

    Example:
        >>> f = cinrad.io.CinradReader(radar_file)
        >>> rl = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
        >>> cappi = cinrad.calc.CAPPI(rl)
        >>> x1d = np.arange(-150000, 150001, 1000)
        >>> y1d = np.arange(-150000, 150001, 1000)
        >>> grid = cappi.get_cappi_xy(x1d, y1d, level_height=3000)
    """

    # 坐标转换常数（与projection.py一致）
    LAT_TO_M = 111000.0  # 1度纬度约111km
    
    def __init__(self, r_list: Volume_T, re: float = 6371000.0, 
                 ke: float = 4.0/3.0, verbose: bool = False):
        from scipy.spatial import cKDTree
        
        self._verbose = verbose
        self._cKDTree = cKDTree
        
        # 去除重复仰角
        seen = set()
        self.rl = [d for d in r_list if d.elevation not in seen and not seen.add(d.elevation)]
        
        self.dtype = get_dtype(r_list[0])
        self.attrs = r_list[0].attrs.copy()
        
        # 雷达位置和仰角信息
        self.radar_lat = self.rl[0].site_latitude
        self.radar_lon = self.rl[0].site_longitude
        self.radar_alt = getattr(self.rl[0], 'radar_height', 0)
        self.elev_angles = np.array([i.elevation for i in self.rl])
        
        # 等效地球半径参数（保留以兼容旧代码）
        self.re = re
        self.ke = ke
        self.reff = ke * re
        
        # 经度转换系数（依赖纬度，与projection.get_coordinate一致）
        self._lon_to_m = self.LAT_TO_M * np.cos(np.deg2rad(self.radar_lat))
        
        # 计算所有数据点的3D坐标（复用scan中已计算的坐标）
        self._build_polar_coords()
        
        # 预过滤有效数据
        valid = ~np.isnan(self.poldata)
        self._valid_coords = self.polcoords[valid]
        self._valid_data = self.poldata[valid]
        
        # KDTree缓存
        self._tree_cache = {}

    def _build_polar_coords(self):
        """计算所有极坐标数据点的3D笛卡尔坐标
        
        复用scan中已计算好的坐标数据：
        - longitude/latitude: 由cinrad.projection.get_coordinate计算
        - height: 由cinrad.projection.height计算
        """
        coords_list, data_list = [], []
        
        for scan in self.rl:
            data = scan[self.dtype].values
            
            # 直接使用scan中已计算好的坐标（由get_coordinate计算）
            lon = scan["longitude"].values
            lat = scan["latitude"].values
            
            # 转换为相对雷达的XY坐标（米）
            x = (lon - self.radar_lon) * self._lon_to_m
            y = (lat - self.radar_lat) * self.LAT_TO_M
            
            # 直接使用scan中已计算好的高度（由projection.height计算，单位km）
            # height数据已包含大气折射校正
            if "height" in scan:
                z = scan["height"].values * 1000  # km -> m
            else:
                # 如果scan中没有height，使用projection.height计算
                from cinrad.projection import height as calc_height
                dist_km = scan["distance"].values
                h_km = calc_height(dist_km, scan.elevation, self.radar_alt)
                n_az = data.shape[0]
                z = np.broadcast_to(h_km * 1000, (n_az, len(dist_km)))
            
            coords_list.append(np.column_stack([x.ravel(), y.ravel(), z.ravel()]))
            data_list.append(data.ravel())
        
        self.polcoords = np.vstack(coords_list)
        self.poldata = np.hstack(data_list)

    def _get_tree(self, height: float, margin: float = 2000.0):
        """获取指定高度范围的KDTree（带缓存）"""
        key = int(height // 500) * 500
        
        if key not in self._tree_cache:
            mask = np.abs(self._valid_coords[:, 2] - height) < margin
            if not mask.any():
                return None, None, 0
            coords, data = self._valid_coords[mask], self._valid_data[mask]
            self._tree_cache[key] = (self._cKDTree(coords), data, len(data))
        
        return self._tree_cache[key]

    def _interpolate(self, grid: np.ndarray, height: float, 
                     skip_mask: np.ndarray = None) -> np.ndarray:
        """3D IDW插值"""
        result = self._get_tree(height)
        if result[0] is None:
            return np.full(len(grid), np.nan)
        
        tree, src_data, n_src = result
        output = np.full(len(grid), np.nan)
        
        # 确定需要插值的点
        idx = np.where(~skip_mask)[0] if skip_mask is not None else np.arange(len(grid))
        if len(idx) == 0:
            return output
        
        # KNN查询 - 使用更大的搜索半径以填补雷达上空空白
        dist_from_center = np.sqrt(grid[idx, 0]**2 + grid[idx, 1]**2)
        # 对于距离雷达中心近的点，使用更大的搜索半径
        max_dist = np.where(dist_from_center < 30000, 15000.0, 8000.0)
        
        # 批量查询
        dists, inds = tree.query(grid[idx], k=6)
        
        # IDW权重计算
        valid = (inds < n_src) & ~np.isinf(dists)
        
        # 对每个点应用距离上限
        for i, md in enumerate(max_dist):
            valid[i] &= (dists[i] < md)
        
        inds = np.where(valid, inds, 0)
        weights = np.where(valid, 1.0 / (dists + 1e-6), 0.0)
        
        w_sum = weights.sum(axis=1)
        has_data = w_sum > 0
        
        if has_data.any():
            norm_w = weights[has_data] / w_sum[has_data, np.newaxis]
            output[idx[has_data]] = (norm_w * src_data[inds[has_data]]).sum(axis=1)
        
        return output

    def _calc_height_at_dist(self, dist_m: np.ndarray, elevation: float) -> np.ndarray:
        """计算指定距离和仰角处的高度（复用projection.height的公式）
        
        使用与projection.height相同的公式：h = r*sin(θ) + r²/(2*RM)
        但接受米为单位的输入，返回米为单位的输出
        """
        # projection.py中 RM = 8500 km，对应 reff = 8500000 m
        # 与 ke * re = 4/3 * 6371000 ≈ 8494667 m 差异<0.1%
        RM_M = 8500000.0  # 与projection.py中RM=8500km一致
        theta = np.deg2rad(elevation)
        return dist_m * np.sin(theta) + dist_m**2 / (2 * RM_M)

    def get_cappi_xy(self, xRange: np.ndarray, yRange: np.ndarray,
                     level_height: float) -> Dataset:
        """生成笛卡尔坐标系的CAPPI

        Args:
            xRange: 东西向坐标数组（米，相对雷达）
            yRange: 南北向坐标数组（米，相对雷达）
            level_height: CAPPI高度（米）

        Returns:
            xarray.Dataset: CAPPI数据
        """
        # 创建网格
        Y, X = np.meshgrid(yRange, xRange, indexing='ij')
        Z = np.full_like(X, level_height)
        grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # 计算盲区
        dist = np.sqrt(grid[:, 0]**2 + grid[:, 1]**2)
        min_el, max_el = self.elev_angles.min(), self.elev_angles.max()
        
        # 高度范围（使用与projection.height一致的公式）
        h_min = self._calc_height_at_dist(dist, min_el)
        h_max = self._calc_height_at_dist(dist, max_el)
        
        # 对于距离雷达较近的区域（<30km），放宽盲区限制
        near_radar = dist < 30000
        skip = np.where(near_radar,
                        False,  # 近雷达区域不跳过
                        (grid[:, 2] < h_min) | (grid[:, 2] > h_max) | (dist > max(abs(xRange.max()), abs(yRange.max())) * 1.5))
        
        # 插值
        cappi = self._interpolate(grid, level_height, skip).reshape(len(yRange), len(xRange))
        
        # 构建结果
        ret = Dataset({self.dtype: DataArray(cappi, coords=[yRange, xRange], dims=["y", "x"])})
        ret.attrs = {**self.attrs, "cappi_height": level_height, 
                     "elevation": level_height / 1000.0}
        return ret

    def get_cappi_lonlat(self, XLon: np.ndarray, YLat: np.ndarray,
                         level_height: float) -> Dataset:
        """生成经纬度坐标系的CAPPI

        Args:
            XLon: 经度数组（度）
            YLat: 纬度数组（度）
            level_height: CAPPI高度（米）

        Returns:
            xarray.Dataset: CAPPI数据
        """
        # 转换为相对坐标（与projection.get_coordinate一致的转换方式）
        xRange = (XLon - self.radar_lon) * self._lon_to_m
        yRange = (YLat - self.radar_lat) * self.LAT_TO_M
        
        # 计算CAPPI
        cappi_xy = self.get_cappi_xy(xRange, yRange, level_height)
        
        # 转换坐标
        ret = Dataset({self.dtype: DataArray(
            cappi_xy[self.dtype].values, coords=[YLat, XLon], dims=["lat", "lon"]
        )})
        ret.attrs = cappi_xy.attrs
        return ret

