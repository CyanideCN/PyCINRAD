# -*- coding: utf-8 -*-
# Author: Puyuan Du

import datetime
import time
from typing import *

import numpy as np
from xarray import DataArray, Dataset

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import KDTree

from cinrad.utils import *
from cinrad.datastruct import Radial, Grid, Slice_
from cinrad.grid import grid_2d, resample
from cinrad.projection import height, get_coordinate
from cinrad.constants import deg2rad
from cinrad.error import RadarCalculationError
from cinrad._typing import Volume_T

__all__ = [
    "quick_cr",
    "quick_et",
    "quick_vil",
    "VCS",
    "quick_vild",
]


def require(var_names: List[str]) -> Callable:
    def wrap(func: Callable) -> Callable:
        def deco(*args, **kwargs) -> Any:
            varset = args[0]
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


def _extract(r_list: List[Dataset], dtype: str) -> tuple:
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


# TODO: Remove later
def is_uniform(radial_list: List[Dataset]) -> bool:
    r"""Check if all input radials have same data type"""
    return len(set([i.dtype for i in radial_list])) == 1


@require(["REF"])
def quick_cr(r_list: List[Dataset], resolution: tuple = (1000, 1000)) -> Dataset:
    r"""
    Calculate composite reflectivity

    Paramters
    ---------
    r_list: list of xarray.Dataset

    Returns
    -------
    l2_obj: xarray.Dataset
        composite reflectivity
    """
    r_data = list()
    for i in r_list:
        r, x, y = grid_2d(
            i["REF"].values,
            i["longitude"].values,
            i["latitude"].values,
            resolution=resolution,
        )
        r_data.append(r)
    cr = np.max(r_data, axis=0)
    x, y = np.meshgrid(x, y)
    # TODO
    l2_obj = Grid(
        np.ma.array(cr, mask=(cr <= 0)),
        i.drange,
        i.reso,
        i.code,
        i.name,
        i.scantime,
        "CR",
        i.stp["lon"],
        i.stp["lat"],
        x,
        y,
        **i.scan_info
    )
    return l2_obj


@require(["REF"])
def quick_et(r_list: List[Dataset]) -> Dataset:
    r"""
    Calculate echo tops

    Paramters
    ---------
    r_list: list of xarray.Dataset

    Returns
    -------
    ret: xarray.Dataset
        echo tops
    """
    r_data, d, a, elev = _extract(r_list, "REF")
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
    r"""
    Calculate vertically integrated liquid

    Paramters
    ---------
    r_list: list of xarray.Dataset

    Returns
    -------
    ret: xarray.Dataset
        vertically integrated liquid
    """
    r_data, d, a, elev = _extract(r_list, "REF")
    i = r_list[0]
    vil = vert_integrated_liquid(
        r_data.astype(np.double), d.astype(np.double), elev.astype(np.double)
    )
    azimuth = a[:, 0]
    distance = d[0]
    ret = Dataset(
        {
            "VIL": DataArray(
                np.ma.masked_less(vil, 0),
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


def quick_vild(r_list: Volume_T) -> Radial:
    r"""
    Calculate vertically integrated liquid density

    Paramters
    ---------
    r_list: list of xarray.Dataset

    Returns
    -------
    l2_obj: xarray.Dataset
        vertically integrated liquid density
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


class VCS(object):
    r"""Class performing vertical cross-section calculation"""

    def __init__(self, r_list: Volume_T):
        if not is_uniform(r_list):
            raise RadarCalculationError(
                "All input radials must have the same data type"
            )
        el = [i.elev for i in r_list]
        if len(el) != len(set(el)):
            self.rl = list()
            el_list = list()
            for data in r_list:
                if data.elev not in el_list:
                    self.rl.append(data)
                    el_list.append(data.elev)
        else:
            self.rl = r_list
        self.x, self.y, self.h, self.r = self._geocoor()

    def _geocoor(self) -> tuple:
        r_data = list()
        x_data = list()
        y_data = list()
        h_data = list()
        for i in self.rl:
            if i.dtype in ["VEL", "SW"]:
                r, x, y = grid_2d(i.data[0], i.lon, i.lat)
            else:
                r, x, y = grid_2d(i.data, i.lon, i.lat)
            r_data.append(r)
            x_data.append(x)
            y_data.append(y)
            hgh_grid, x, y = grid_2d(i.height, i.lon, i.lat)
            h_data.append(hgh_grid)
        return x_data, y_data, h_data, r_data

    def _get_section(
        self, stp: Tuple[float, float], enp: Tuple[float, float], spacing: int
    ) -> Slice_:
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
        r = np.ma.masked_invalid(r)
        x = np.linspace(0, 1, spacing) * np.ones(r.shape[0])[:, np.newaxis]
        stp_s = "{}N, {}E".format(stp[1], stp[0])
        enp_s = "{}N, {}E".format(enp[1], enp[0])
        sl = Slice_(
            r,
            x,
            h,
            self.rl[0].scantime,
            self.rl[0].code,
            self.rl[0].name,
            self.rl[0].dtype,
            stp_s=stp_s,
            enp_s=enp_s,
            stp=stp,
            enp=enp,
        )
        return sl

    def get_section(
        self,
        start_polar: Optional[Tuple[float, float]] = None,
        end_polar: Optional[Tuple[float, float]] = None,
        start_cart: Optional[Tuple[float, float]] = None,
        end_cart: Optional[Tuple[float, float]] = None,
        spacing: int = 500,
    ) -> Slice_:
        r"""
        Get cross-section data from input points

        Parameters
        ----------
        start_polar: list or tuple
            polar coordinates of start point i.e.(distance, azimuth)
        end_polar: list or tuple
            polar coordinates of end point i.e.(distance, azimuth)
        start_cart: list or tuple
            geographic coordinates of start point i.e.(longitude, latitude)
        end_cart: list or tuple
            geographic coordinates of end point i.e.(longitude, latitude)

        Returns
        -------
        sl: cinrad.datastruct.Slice_
        """
        if start_polar and end_polar:
            stlat = self.rl[0].stp["lat"]
            stlon = self.rl[0].stp["lon"]
            stp = np.round_(
                get_coordinate(
                    start_polar[0], start_polar[1] * deg2rad, 0, stlon, stlat
                ),
                2,
            )
            enp = np.round_(
                get_coordinate(end_polar[0], end_polar[1] * deg2rad, 0, stlon, stlat), 2
            )
        elif start_cart and end_cart:
            stp = start_cart
            enp = end_cart
        else:
            raise RadarCalculationError("Invalid input")
        return self._get_section(stp, enp, spacing)


class GridMapper(object):
    def __init__(self, fields: Volume_T, max_dist: Number_T = 0.1):
        # Process data type
        if len(set([i.dtype for i in fields])) > 1:
            raise RadarCalculationError("All input data should have same data type")
        self.dtype = fields[0].dtype
        # Process time
        t_arr = np.array([time.mktime(i.scantime.timetuple()) for i in fields])
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
        self.lon_ravel = np.hstack([i.lon.ravel() for i in fields])
        self.lat_ravel = np.hstack([i.lat.ravel() for i in fields])
        self.data_ravel = np.ma.hstack([i.data.ravel() for i in fields])
        self.dist_ravel = np.hstack(
            [np.broadcast_to(i.dist, i.lon.shape).ravel() for i in fields]
        )
        self.tree = KDTree(np.dstack((self.lon_ravel, self.lat_ravel))[0])
        self.md = max_dist

    def _process_grid(self, x_step: Number_T, y_step: Number_T) -> Tuple[np.ndarray]:
        x_lower = np.round_(self.lon_ravel.min(), 2)
        x_upper = np.round_(self.lon_ravel.max(), 2)
        y_lower = np.round_(self.lat_ravel.min(), 2)
        y_upper = np.round_(self.lat_ravel.max(), 2)
        x_grid = np.arange(x_lower, x_upper + x_step, x_step)
        y_grid = np.arange(y_lower, y_upper + x_step, x_step)
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

    def __call__(self, step: Number_T) -> Grid:
        x, y = self._process_grid(step, step)
        grid = self._map_points(x, y)
        grid = np.ma.masked_outside(grid, 0.1, 100)
        return Grid(
            grid,
            np.nan,
            step,
            "RADMAP",
            "RADMAP",
            self.scan_time,
            self.dtype,
            0,
            0,
            x,
            y,
        )
