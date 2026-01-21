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
    Class performing CAPPI (Constant Altitude Plan Position Indicator) calculation
    Based on wradlib's approach with 3D voxel interpolation.

    Supports optional Earth curvature and atmospheric refraction correction using
    the 4/3 Earth radius model (or custom values via re and ke parameters).

    Args:
        r_list (list(xarray.Dataset)): List of reflectivity data from each elevation angle.
        re (float): Effective Earth radius in meters. Default 6371000 (standard Earth radius).
                     For standard atmospheric refraction, use 4/3 * 6371000 ≈ 8480000.
        ke (float): Refraction coefficient. Default 4/3 for standard atmosphere.
                     Set to 1.0 to disable refraction correction.
        wradlib_mode (bool): If True, use wradlib.spherical_to_proj for coordinate conversion
                            (requires wradlib). If False, use native implementation.
                            Default: False (use native implementation).

    Example:
        >>> f = cinrad.io.CinradReader(radar_file)
        >>> rl = [f.get_data(i, 230, 'REF') for i in f.angleindex_r]
        >>> cappi = cinrad.calc.CAPPI(rl)

        Basic usage (standard 4/3 Earth radius model):
        >>> x1d = np.arange(-150000, 150001, 1000)
        >>> y1d = np.arange(-150000, 150001, 1000)
        >>> grid = cappi.get_cappi_xy(x1d, y1d, level_height=3000)

        With custom Earth curvature parameters:
        >>> cappi = cinrad.calc.CAPPI(rl, re=6371000, ke=1.0)  # No refraction
        >>> cappi = cinrad.calc.CAPPI(rl, re=8480000, ke=4/3)  # Explicit 4/3 model

        Geographic coordinates:
        >>> lon1d = np.arange(117, 120.001, 0.01)
        >>> lat1d = np.arange(31, 34.001, 0.01)
        >>> grid_geo = cappi.get_cappi_lonlat(lon1d, lat1d, level_height=3000)
    """

    def __init__(self, r_list: Volume_T, re: float = None, ke: float = None, 
                 wradlib_mode: bool = False, verbose: bool = False):
        """Initialize CAPPI with optional performance timing.
        
        Args:
            r_list: List of radar data from each elevation angle
            re: Earth radius in meters (default: 6371000)
            ke: Refraction coefficient (default: 4/3)
            wradlib_mode: Use wradlib for coordinate conversion
            verbose: Print timing information for performance analysis
        """
        import time as _time
        self._verbose = verbose
        t0 = _time.perf_counter()
        
        # Remove duplicate elevation angles
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
        self.attrs = r_list[0].attrs

        # Get radar location
        self.radar_lat = self.rl[0].site_latitude
        self.radar_lon = self.rl[0].site_longitude
        self.radar_alt = getattr(self.rl[0], 'radar_height', 0)
        self.elev_angles = np.array([i.elevation for i in self.rl])

        # Earth curvature and refraction parameters
        self.re = re if re is not None else 6371000.0
        self.ke = ke if ke is not None else 4.0 / 3.0
        self.reff = self.ke * self.re

        # Try to import wradlib if requested
        self.wradlib_mode = wradlib_mode
        self._wradlib = None
        if wradlib_mode:
            try:
                from wradlib.georef import polar as wradlib_polar
                self._wradlib = wradlib_polar
            except ImportError:
                import warnings
                warnings.warn(
                    "wradlib not found, falling back to native implementation."
                )
                self.wradlib_mode = False

        # KDTree cache for different height levels
        self._kdtree_cache = {}
        self._filtered_data_cache = {}
        
        # Calculate 3D polar coordinates for all data points
        self._calculate_polar_coords()
        
        # Pre-filter valid data (non-NaN) once
        self._valid_mask = ~np.isnan(self.poldata)
        self._valid_coords = self.polcoords[self._valid_mask]
        self._valid_data = self.poldata[self._valid_mask]
        
        if self._verbose:
            print(f"[CAPPI] Init time: {(_time.perf_counter()-t0)*1000:.1f}ms, "
                  f"valid points: {len(self._valid_data)}/{len(self.poldata)}")

    def _calculate_polar_coords(self):
        """Calculate 3D Cartesian coordinates for all polar data points.
        
        Uses the 4/3 Earth radius model for atmospheric refraction correction.
        Height calculation follows wradlib's spherical_to_proj approach:
        
            z = r * sin(θ) + r² / (2 * R_eff)
            
        where R_eff = ke * re is the effective Earth radius.
        
        For x, y coordinates: uses longitude/latitude conversion with proper
        projection to account for Earth's curvature.
        """
        
        all_coords = []
        all_data = []
        
        for scan in self.rl:
            data = scan[self.dtype].values
            
            # Get polar coordinates from scan
            lon = scan["longitude"].values
            lat = scan["latitude"].values
            distance = scan["distance"].values  # km, shape (n_distance,)
            azimuth = scan["azimuth"].values if "azimuth" in scan.coords else None
            elevation = scan.elevation
            
            # Get data shape for broadcasting
            n_azimuth, n_distance = data.shape
            
            # Use wradlib if available and requested
            if self.wradlib_mode and self._wradlib is not None:
                # wradlib spherical_to_proj
                # Note: wradlib expects distance in meters
                r = distance * 1000  # Convert km to m
                theta = np.full_like(r, elevation) if azimuth is None else elevation
                site = (self.radar_lon, self.radar_lat, self.radar_alt)
                
                coords = self._wradlib.spherical_to_proj(r, azimuth, theta, site)
                # coords shape: (n_azimuth, n_range, 3) -> (..., 3) with lon, lat, z
                # Flatten to (n_points, 3)
                coords_3d = coords.reshape(-1, 3)
                
                # Convert to x, y, z relative to radar (in meters)
                lat_to_m = 111000.0
                lon_to_m = 111000.0 * np.cos(np.deg2rad(self.radar_lat))
                
                x_coords = (coords_3d[:, 0] - self.radar_lon) * lon_to_m
                y_coords = (coords_3d[:, 1] - self.radar_lat) * lat_to_m
                z_coords = coords_3d[:, 2]
            else:
                # Native implementation with 4/3 Earth radius model
                
                # Calculate x, y from longitude/latitude (meters from radar)
                lat_to_m = 111000.0
                lon_to_m = 111000.0 * np.cos(np.deg2rad(self.radar_lat))
                
                x_coords = (lon - self.radar_lon) * lon_to_m
                y_coords = (lat - self.radar_lat) * lat_to_m
                
                # Calculate height using 4/3 Earth radius model
                # Beam height: h = r * sin(θ) + r² / (2 * R_eff)
                # where r is slant range, θ is elevation angle
                # R_eff = ke * re is effective Earth radius
                
                r = distance * 1000  # Convert km to m, shape (n_distance,)
                theta_rad = np.deg2rad(elevation)
                
                # Beam height above radar: shape (n_distance,)
                h_beam_1d = r * np.sin(theta_rad) + (r ** 2) / (2 * self.reff)
                
                # Broadcast to match data shape (n_azimuth, n_distance)
                h_beam = np.broadcast_to(h_beam_1d, (n_azimuth, n_distance))
                z_coords = h_beam + self.radar_alt
            
            # Flatten and store
            coords = np.column_stack([
                x_coords.ravel(),
                y_coords.ravel(),
                z_coords.ravel()
            ])
            
            all_coords.append(coords)
            all_data.append(data.ravel())
        
        # Combine all sweeps
        self.polcoords = np.vstack(all_coords)
        self.poldata = np.hstack(all_data)
        
        # Store shape info for later reshaping
        self.polshape = (len(self.rl), 
                        self.rl[0][self.dtype].shape[0], 
                        self.rl[0][self.dtype].shape[1])

    def _make_3d_grid(self, x_range: np.ndarray, y_range: np.ndarray, 
                      z_levels: np.ndarray) -> tuple:
        """Create 3D Cartesian grid coordinates.
        
        Args:
            x_range: x coordinates (meters from radar)
            y_range: y coordinates (meters from radar)
            z_levels: z coordinates (altitude in meters)
            
        Returns:
            gridcoords: Array of shape (num_voxels, 3)
            gridshape: Tuple (nz, ny, nx) representing the grid shape
        """
        # Create 3D meshgrid
        Z, Y, X = np.meshgrid(z_levels, y_range, x_range, indexing='ij')
        
        # Flatten to get voxel coordinates
        gridcoords = np.column_stack([
            X.ravel(),
            Y.ravel(),
            Z.ravel()
        ])
        
        gridshape = (len(z_levels), len(y_range), len(x_range))
        
        return gridcoords, gridshape

    def _calculate_blindspots(self, gridcoords: np.ndarray, 
                                minelev: float, maxelev: float, 
                                maxrange: float) -> tuple:
        """Calculate blind spots for the CAPPI.
        
        Blind spots are grid points that are:
        1. Below the lowest elevation angle (considering Earth curvature)
        2. Above the highest elevation angle  
        3. Beyond the maximum range
        
        Uses the same 4/3 Earth radius model for consistency.
        
        Args:
            gridcoords: Array of shape (num_voxels, 3)
            minelev: Minimum elevation angle (degrees)
            maxelev: Maximum elevation angle (degrees)
            maxrange: Maximum range (meters)
            
        Returns:
            tuple: (below, above, out_of_range) boolean arrays
        """
        # Distance from radar (handle NaN values)
        dxdy = np.abs(gridcoords[:, :2])  # Use absolute values
        dist = np.sqrt((dxdy ** 2).sum(axis=1))
        
        # Set NaN distances to infinity so they are masked
        dist = np.where(np.isnan(dist), np.inf, dist)
        
        # Below the radar (below min elevation at that distance)
        # Using 4/3 Earth radius model: h = r * sin(el) + r² / (2 * R_eff)
        h_min = dist * np.sin(np.deg2rad(minelev)) + dist**2 / (2 * self.reff)
        below = (gridcoords[:, 2] < h_min) | np.isnan(gridcoords[:, 2])
        
        # Above the radar (above max elevation at that distance)
        h_max = dist * np.sin(np.deg2rad(maxelev)) + dist**2 / (2 * self.reff)
        above = (gridcoords[:, 2] > h_max) | np.isnan(gridcoords[:, 2])
        
        # Out of range
        out_of_range = (dist > maxrange) | np.isnan(dist)
        
        return below, above, out_of_range

    def _get_cached_tree(self, target_height: float, v_margin: float = 1500.0):
        """Get or build cached KDTree for a height range.
        
        Uses height binning to maximize cache hits (500m bins).
        """
        from scipy.spatial import cKDTree
        
        # Round height to 500m bins for cache key
        cache_key = int(target_height // 500) * 500
        
        if cache_key in self._kdtree_cache:
            return self._kdtree_cache[cache_key]
        
        # Filter by height range using pre-filtered valid data
        z_dist = np.abs(self._valid_coords[:, 2] - target_height)
        height_mask = z_dist < v_margin
        
        if not np.any(height_mask):
            return None, None, 0
        
        src_coords = self._valid_coords[height_mask]
        src_data = self._valid_data[height_mask]
        
        # Build and cache tree
        tree = cKDTree(src_coords)
        self._kdtree_cache[cache_key] = (tree, src_data, len(src_coords))
        self._filtered_data_cache[cache_key] = src_data
        
        return tree, src_data, len(src_coords)

    def _interpolate_3d(self, gridcoords: np.ndarray, 
                        target_height: float,
                        mask: np.ndarray = None,
                        v_margin: float = 1500.0) -> np.ndarray:
        """Optimized 3D interpolation with KDTree caching.
        
        Args:
            gridcoords: Target grid coordinates (N, 3)
            target_height: Target CAPPI height in meters
            mask: Boolean mask for blind spots (True = skip)
            v_margin: Vertical margin for data filtering (meters)
        """
        import time as _time
        t0 = _time.perf_counter()
        
        # 1. Get cached tree or build new one
        result = self._get_cached_tree(target_height, v_margin)
        if result[0] is None:
            return np.full(len(gridcoords), np.nan)
        
        tree, src_data, n_src = result
        t1 = _time.perf_counter()
        
        # 2. Prepare target points (skip masked)
        trg_indices = np.where(~mask)[0] if mask is not None else np.arange(len(gridcoords))
        output = np.full(len(gridcoords), np.nan)
        
        if len(trg_indices) == 0:
            return output

        # 3. Batch KNN query (k=4 is sufficient for IDW)
        dists, indices = tree.query(gridcoords[trg_indices], k=4, distance_upper_bound=5000)
        t2 = _time.perf_counter()
        
        # 4. Vectorized IDW interpolation
        invalid_mask = (indices >= n_src)
        indices[invalid_mask] = 0
        
        weights = 1.0 / (dists + 1e-6)
        weights[invalid_mask] = 0
        weights[np.isinf(dists)] = 0
        
        sum_weights = np.sum(weights, axis=1)
        valid_trg_mask = sum_weights > 0
        
        if not np.any(valid_trg_mask):
            return output
        
        neighbor_data = src_data[indices[valid_trg_mask]]
        norm_weights = weights[valid_trg_mask] / sum_weights[valid_trg_mask][:, np.newaxis]
        interpolated_vals = np.sum(norm_weights * neighbor_data, axis=1)
        
        actual_indices = trg_indices[valid_trg_mask]
        output[actual_indices] = interpolated_vals
        
        if self._verbose:
            print(f"[CAPPI] Interpolate: tree={1000*(t1-t0):.1f}ms, "
                  f"query={1000*(t2-t1):.1f}ms, total={1000*(_time.perf_counter()-t0):.1f}ms")
        
        return output

    def get_cappi_xy(self, xRange: np.ndarray, yRange: np.ndarray,
                      level_height: float) -> Dataset:
        r"""
        Generate CAPPI in cartesian coordinates centered on radar.
        Based on wradlib's 3D voxel interpolation approach.

        Args:
            xRange (numpy.ndarray): East-west coordinate array in meters from radar.
            yRange (numpy.ndarray): North-south coordinate array in meters from radar.
            level_height (float): Target height for CAPPI in meters.

        Returns:
            xarray.Dataset: CAPPI data with coordinates (x, y) in meters.
        """
        from xarray import DataArray, Dataset
        
        # Create 3D grid with single height level
        z_levels = np.array([level_height])
        gridcoords, gridshape = self._make_3d_grid(xRange, yRange, z_levels)
        
        # Calculate blind spots
        minelev = self.elev_angles.min()
        maxelev = self.elev_angles.max()
        maxrange = xRange.max() * 1.5  # Allow some margin
        
        below, above, out_of_range = self._calculate_blindspots(
            gridcoords, minelev, maxelev, maxrange
        )
        
        # Mask for blind voxels
        mask = below | above | out_of_range
        
        # Perform 3D interpolation
        gridded = self._interpolate_3d(gridcoords, level_height, mask)
        
        # Reshape to 2D
        cappi_2d = gridded.reshape(gridshape[1], gridshape[2])
        
        # Create output Dataset
        ret = Dataset({
            self.dtype: DataArray(
                cappi_2d,
                coords=[yRange, xRange],
                dims=["y", "x"]
            )
        })

        # Set attributes
        r_attr = self.attrs.copy()
        r_attr["cappi_height"] = level_height
        r_attr["elevation"] = level_height / 1000.0
        r_attr["tangential_reso"] = np.nan
        r_attr["range"] = np.nan

        ret.attrs = r_attr

        return ret

    def get_cappi_lonlat(self, XLon: np.ndarray, YLat: np.ndarray,
                          level_height: float) -> Dataset:
        r"""
        Generate CAPPI in geographic coordinates (longitude, latitude).
        Based on wradlib's 3D voxel interpolation approach.

        Args:
            XLon (numpy.ndarray): Longitude array in degrees.
            YLat (numpy.ndarray): Latitude array in degrees.
            level_height (float): Target height for CAPPI in meters.

        Returns:
            xarray.Dataset: CAPPI data with coordinates (lon, lat) in degrees.
        """
        from xarray import DataArray, Dataset
        
        # Convert lon/lat to x/y relative to radar
        # 1 degree of longitude ≈ 111km * cos(latitude)
        # 1 degree of latitude ≈ 111km
        
        lat_rad = np.deg2rad(self.radar_lat)
        lon_to_m = 111000.0 * np.cos(lat_rad)
        lat_to_m = 111000.0
        
        xRange = (XLon - self.radar_lon) * lon_to_m
        yRange = (YLat - self.radar_lat) * lat_to_m
        
        # Generate CAPPI using xy coordinates
        cappi_xy = self.get_cappi_xy(xRange, yRange, level_height)
        
        # Create output with lon/lat coordinates
        ret = Dataset({
            self.dtype: DataArray(
                cappi_xy[self.dtype].values,
                coords=[YLat, XLon],
                dims=["lat", "lon"]
            )
        })

        # Set attributes
        r_attr = self.attrs.copy()
        r_attr["cappi_height"] = level_height
        r_attr["elevation"] = level_height / 1000.0
        r_attr["tangential_reso"] = np.nan
        r_attr["range"] = np.nan

        ret.attrs = r_attr

        return ret
