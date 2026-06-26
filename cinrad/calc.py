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
from cinrad.cappi_numba import HAS_NUMBA, cappi_sweep_interp

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

    # Get grid from the first tilt
    r, x, y = grid_2d(
        r_list[0]["REF"].values,
        r_list[0]["longitude"].values,
        r_list[0]["latitude"].values,
        resolution=resolution,
    )
    cr = r
    for i in r_list[1:]:
        r, _, _ = grid_2d(
            i["REF"].values,
            i["longitude"].values,
            i["latitude"].values,
            x_out=x,
            y_out=y,
            resolution=resolution,
        )
        cr = np.fmax(cr, r)
    ret = Dataset({"CR": DataArray(cr, coords=[y, x], dims=["latitude", "longitude"])})
    ret.attrs = r_list[0].attrs.copy()
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
        grid0 = self.fields[0].interp(longitude=x[0], latitude=y[:, 0], method="linear")
        r_data_max = grid0[self.dtype].values
        for field in self.fields[1:]:
            field_grid = field.interp(longitude=x[0], latitude=y[:, 0], method="linear")
            current = field_grid[self.dtype].values
            r_data_max = np.fmax(r_data_max, current)
       
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
    CAPPI (Constant Altitude Plan Position Indicator) via sweep-based interpolation.

    采用与 pycwr 等价的 sweep-based 插值算法：
    1. 对每个网格点，使用球面几何将 (x, y, z) 反算为天线坐标 (azimuth, range, elevation)
    2. 根据仰角找到上下两个仰角层
    3. 在每个仰角层内做 (azimuth, range) 双线性插值（含方位角 0/360 循环处理）
    4. 在仰角层间做 elevation 方向线性插值

    Args:
        r_list: 各仰角的反射率数据列表 (Volume_T)
        re: 地球半径 (m)，默认 6371000
        ke: 等效地球半径因子，默认 4/3
        verbose: 是否打印性能信息

    Example:
        >>> f = cinrad.io.CinradReader(radar_file)
        >>> rl = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
        >>> cappi = cinrad.calc.CAPPI(rl)
        >>> x1d = np.arange(-150000, 150001, 1000)
        >>> y1d = np.arange(-150000, 150001, 1000)
        >>> grid = cappi.get_cappi_xy(x1d, y1d, level_height=3000)
    """

    # 等效地球半径 4/3 × 6371000 (m)，与 pycwr 一致
    REFF = 8494666.666666666

    def __init__(self, r_list: Volume_T, re: float = 6371000.0,
                 ke: float = 4.0 / 3.0, verbose: bool = False):
        self._verbose = verbose

        # 去除重复仰角
        seen = set()
        self.rl = [d for d in r_list
                   if d.elevation not in seen and not seen.add(d.elevation)]

        self.dtype = get_dtype(r_list[0])
        self.attrs = r_list[0].attrs.copy()

        # 雷达位置
        self.radar_lat = self.rl[0].site_latitude
        self.radar_lon = self.rl[0].site_longitude

        # 等效地球半径参数（用于 _cartesian_to_antenna 球面几何计算）
        self.reff = ke * re  # ≈ 8494666.67 m

        # 仰角数组
        self.elev_angles = np.array([i.elevation for i in self.rl],
                                    dtype=np.float64)

        # 提取各仰角层的 (azimuth, range, data)
        # azimuth 单位: 弧度 (与 Dataset 保持一致)
        # range 单位: km (与 Dataset 保持一致)
        self._sweep_azimuths = []  # list of 1D ndarray (rad)
        self._sweep_ranges = []    # list of 1D ndarray (km)
        self._sweep_data = []      # list of 2D ndarray (naz, nrange)

        for scan in self.rl:
            data = scan[self.dtype].values
            az = scan["azimuth"].values       # radians, sorted
            r = scan["distance"].values       # km
            self._sweep_azimuths.append(
                np.ascontiguousarray(az, dtype=np.float64))
            self._sweep_ranges.append(
                np.ascontiguousarray(r, dtype=np.float64))
            self._sweep_data.append(
                np.ascontiguousarray(data, dtype=np.float64))

        # 雷达天线海拔高度 (m MSL)
        self.radar_height = 0.0
        try:
            # 从 ds["height"] 首栅格反推:
            # h = d*sin(el) + d²/(2*reff) + h_radar/1000  (projection.height 公式)
            # → 反解 h_radar = (h - d*sin(el) - d²/(2*reff)) * 1000
            h_km = float(self.rl[0]["height"].values[0, 0])  # km
            d_km = float(self._sweep_ranges[0][0])           # km
            el = self.elev_angles[0]
            h_radar = (h_km - d_km * np.sin(np.deg2rad(el))
                       - d_km**2 / (2.0 * self.reff / 1000.0)) * 1000.0
            self.radar_height = max(0.0, float(h_radar))
        except (KeyError, IndexError, ValueError, TypeError):
            self.radar_height = 0.0

    # ------------------------------------------------------------------
    # 坐标转换：球面几何反算 Cartesian → Antenna
    # 等价于 pycwr 的 _cartesian_to_antenna_impl
    # ------------------------------------------------------------------
    @staticmethod
    def _cartesian_to_antenna(x, y, z, h_radar=0.0, reff=REFF):
        """
        使用球面几何反算天线坐标（全矢量版本）。

        Parameters
        ----------
        x, y : array_like, 相对雷达的笛卡尔坐标 (m)
        z : array_like, 目标高度 (m MSL)
        h_radar : float, 雷达高度 (m MSL)
        reff : float, 等效地球半径 (m)

        Returns
        -------
        azimuth : ndarray, 方位角 (deg, [0, 360))
        ranges : ndarray, 斜距 (m)
        elevation : ndarray, 仰角 (deg)
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        s = np.sqrt(x**2 + y**2)
        phi = s / reff
        radar_radius = reff + h_radar
        gate_radius = reff + z

        # 斜距
        ranges_sq = (radar_radius**2 + gate_radius**2 -
                     2.0 * radar_radius * gate_radius * np.cos(phi))
        ranges_sq = np.maximum(ranges_sq, 0.0)
        ranges = np.sqrt(ranges_sq)

        # 仰角
        cos_arg = np.clip(
            (radar_radius**2 + ranges**2 - gate_radius**2) /
            (2.0 * radar_radius * np.maximum(ranges, 1e-10)),
            -1.0, 1.0,
        )
        elevation = np.rad2deg(np.arccos(cos_arg) - np.pi / 2.0)
        elevation[ranges < 1e-10] = 0.0

        # 方位角 (0~360 deg, 数学角: 北=0 顺时针)
        azimuth = np.rad2deg(np.pi / 2.0 - np.arctan2(y, x))
        azimuth = np.mod(azimuth, 360.0)

        return azimuth, ranges, elevation

    # ------------------------------------------------------------------
    # 线性插值（等价于 pycwr 的 _interp_linear）
    # ------------------------------------------------------------------
    @staticmethod
    def _interp_linear(coord, coord_0, coord_1, dat_0, dat_1, fillvalue):
        """一维线性插值，处理 NaN。"""
        if coord_1 == coord_0:
            return fillvalue
        n0 = np.isnan(dat_0)
        n1 = np.isnan(dat_1)
        if n0 and n1:
            return fillvalue
        if n0:
            return dat_1
        if n1:
            return dat_0
        return ((coord_1 - coord) * dat_0 +
                (coord - coord_0) * dat_1) / (coord_1 - coord_0)

    # ------------------------------------------------------------------
    # 单仰角层内 (azimuth, range) 双线性插值
    # 等价于 pycwr 的 _interp_sweep_sample
    # ------------------------------------------------------------------
    def _interp_sweep_sample(self, sweep_idx, target_az, target_r, fillvalue):
        """
        在指定仰角层内做 (azimuth, range) 双线性插值。

        Parameters
        ----------
        sweep_idx : int, 仰角层索引
        target_az : float, 目标方位角 (deg)
        target_r : float, 目标斜距 (m)
        fillvalue : float, 填充值

        Returns
        -------
        float, 插值结果
        """
        azimuth = self._sweep_azimuths[sweep_idx]   # rad, sorted
        ranges = self._sweep_ranges[sweep_idx]       # km, sorted
        data = self._sweep_data[sweep_idx]           # (naz, nrange)

        naz = len(azimuth)
        nrange = len(ranges)

        if naz < 2 or nrange < 2:
            return fillvalue

        r_km = target_r / 1000.0  # m → km

        # 超出距离范围
        if r_km > ranges[-1] or r_km < ranges[0]:
            return fillvalue

        # --- 查找方位角索引 ---
        az_rad = np.deg2rad(target_az)
        iaz = np.searchsorted(azimuth, az_rad, side='right')

        if iaz >= naz:
            # 超过最大方位角 → 绕回 0
            iaz = 0
            az_rad_val = az_rad - 2.0 * np.pi
        else:
            az_rad_val = az_rad

        if iaz == 0:
            # 低于最小方位角 → 用最后一个方位角 - 360 作为下界
            az_last = azimuth[naz - 1] - 2.0 * np.pi
        else:
            az_last = azimuth[iaz - 1]

        # --- 查找距离索引 ---
        ir = np.searchsorted(ranges, r_km, side='right')
        if ir <= 0:
            ir = 1
        elif ir >= nrange:
            ir = nrange - 1

        # --- 双线性插值（与 pycwr 完全一致） ---
        er0 = self._interp_linear(
            az_rad_val, az_last, azimuth[iaz],
            data[iaz - 1, ir - 1], data[iaz, ir - 1], fillvalue)
        er1 = self._interp_linear(
            az_rad_val, az_last, azimuth[iaz],
            data[iaz - 1, ir], data[iaz, ir], fillvalue)
        return self._interp_linear(
            r_km, ranges[ir - 1], ranges[ir], er0, er1, fillvalue)

    # ------------------------------------------------------------------
    # 主入口：笛卡尔坐标 CAPPI
    # 等价于 pycwr 的 get_CAPPI_xy
    # ------------------------------------------------------------------
    def get_cappi_xy(self, xRange: np.ndarray, yRange: np.ndarray,
                     level_height: float, fillvalue=np.nan,
                     beam_width_deg: float = 1.0) -> Dataset:
        """
        生成笛卡尔坐标系的 CAPPI。

        使用 sweep-based 插值算法，等价于 pycwr 的 get_CAPPI_xy。

        Parameters
        ----------
        xRange : 1D array, 东西向坐标 (m，相对雷达)
        yRange : 1D array, 南北向坐标 (m，相对雷达)
        level_height : float, CAPPI 高度 (m MSL)
        fillvalue : float, 填充值 (默认 NaN)
        beam_width_deg : float, 波束宽度 (度)

        Returns
        -------
        xarray.Dataset
        """
        Y, X = np.meshgrid(yRange, xRange, indexing='ij')
        ny, nx = Y.shape
        n_total = ny * nx

        # 1. 将网格点坐标反算为天线坐标 (球面几何，全矢量)
        t0 = time.time()
        az_deg, r_m, el_deg = self._cartesian_to_antenna(
            X.ravel(), Y.ravel(),
            np.full(n_total, level_height, dtype=np.float64),
            self.radar_height, self.reff)
        if self._verbose:
            print(f"[CAPPI] 坐标转换耗时: {time.time() - t0:.2f}s")

        # 2. 对每个网格点做 sweep-based 插值
        t1 = time.time()
        beam_half = max(beam_width_deg * 0.5, 0.1)

        if HAS_NUMBA:
            # numba 并行加速路径（~20-50×）
            from cinrad.cappi_numba import cappi_sweep_interp
            result = cappi_sweep_interp(
                X.ravel(), Y.ravel(),
                level_height, self.radar_height, self.reff,
                self._sweep_azimuths, self._sweep_ranges, self._sweep_data,
                self.elev_angles,
                fillvalue=fillvalue, beam_half=beam_half,
            )
        else:
            # 纯 Python 路径
            ne = len(self.elev_angles)
            result = np.full(n_total, fillvalue, dtype=np.float64)

            for i in range(n_total):
                el = el_deg[i]
                az = az_deg[i]
                r = r_m[i]

                if np.isnan(el) or np.isnan(r) or r <= 0:
                    continue

                if ne == 1:
                    if abs(el - self.elev_angles[0]) > beam_half:
                        continue
                    result[i] = self._interp_sweep_sample(
                        0, az, r, fillvalue)
                    continue

                if el < self.elev_angles[0] - beam_half:
                    continue
                if el > self.elev_angles[-1] + beam_half:
                    continue

                if el <= self.elev_angles[0] + beam_half:
                    result[i] = self._interp_sweep_sample(
                        0, az, r, fillvalue)
                elif el >= self.elev_angles[-1] - beam_half:
                    result[i] = self._interp_sweep_sample(
                        ne - 1, az, r, fillvalue)
                else:
                    ie = np.searchsorted(self.elev_angles, el, side='right')
                    if ie >= ne:
                        ie = ne - 1
                    lower = ie - 1
                    upper = ie

                    val_low = self._interp_sweep_sample(
                        lower, az, r, fillvalue)
                    val_up = self._interp_sweep_sample(
                        upper, az, r, fillvalue)

                    if np.isnan(val_low) and np.isnan(val_up):
                        result[i] = fillvalue
                    elif np.isnan(val_low):
                        result[i] = val_up
                    elif np.isnan(val_up):
                        result[i] = val_low
                    else:
                        w = ((el - self.elev_angles[lower]) /
                             (self.elev_angles[upper] -
                              self.elev_angles[lower]))
                        result[i] = val_low * (1.0 - w) + val_up * w

        if self._verbose:
            print(f"[CAPPI] 插值耗时: {time.time() - t1:.2f}s")

        result_2d = result.reshape(ny, nx)
        ret = Dataset({
            self.dtype:
                DataArray(result_2d, coords=[yRange, xRange], dims=["y", "x"])
        })
        ret.attrs = {**self.attrs,
                     "cappi_height": level_height,
                     "elevation": level_height / 1000.0}
        return ret

    def get_cappi_lonlat(self, XLon: np.ndarray, YLat: np.ndarray,
                         level_height: float, fillvalue=np.nan,
                         beam_width_deg: float = 1.0) -> Dataset:
        """
        生成经纬度坐标系的 CAPPI。

        Parameters
        ----------
        XLon : array, 经度 (度)
        YLat : array, 纬度 (度)
        level_height : float, CAPPI 高度 (m MSL)
        fillvalue : float, 填充值
        beam_width_deg : float, 波束宽度 (度)

        Returns
        -------
        xarray.Dataset
        """
        LAT_TO_M = 111000.0
        _lon_to_m = LAT_TO_M * np.cos(np.deg2rad(self.radar_lat))

        xRange = (XLon - self.radar_lon) * _lon_to_m
        yRange = (YLat - self.radar_lat) * LAT_TO_M

        cappi_xy = self.get_cappi_xy(xRange, yRange, level_height,
                                      fillvalue=fillvalue,
                                      beam_width_deg=beam_width_deg)

        ret = Dataset({
            self.dtype: DataArray(cappi_xy[self.dtype].values,
                                  coords=[YLat, XLon],
                                  dims=["lat", "lon"])
        })
        ret.attrs = cappi_xy.attrs
        return ret
