# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import time

import numpy as np
from xarray import DataArray, Dataset

from cinrad._typing import Volume_T
from cinrad.cappi_numba import HAS_NUMBA
from cinrad.common import get_dtype


def _fill_is_nan(fillvalue) -> bool:
    try:
        return bool(np.isnan(fillvalue))
    except (TypeError, ValueError):
        return False


def _replace_fillvalue(data: np.ndarray, fillvalue):
    if _fill_is_nan(fillvalue):
        return data
    return np.where(np.isnan(data), fillvalue, data)


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
        if not r_list:
            raise ValueError("Input radar volume cannot be empty")

        seen = set()
        self.rl = sorted(
            [d for d in r_list
             if d.elevation not in seen and not seen.add(d.elevation)],
            key=lambda d: d.elevation,
        )

        self.dtype = get_dtype(self.rl[0])
        self.attrs = self.rl[0].attrs.copy()

        self.radar_lat = self.rl[0].site_latitude
        self.radar_lon = self.rl[0].site_longitude
        self.reff = ke * re
        self.elev_angles = np.array([i.elevation for i in self.rl],
                                    dtype=np.float64)

        self._sweep_azimuths = []
        self._sweep_ranges = []
        self._sweep_data = []
        self._prepare_sweep_data()
        self.radar_height = self._resolve_radar_height()

    def _prepare_sweep_data(self):
        for scan in self.rl:
            data = np.asarray(scan[self.dtype].values, dtype=np.float64)
            az = np.mod(np.asarray(scan["azimuth"].values,
                                   dtype=np.float64), 2.0 * np.pi)
            r = np.asarray(scan["distance"].values, dtype=np.float64)

            az_order = np.argsort(az, kind="mergesort")
            r_order = np.argsort(r, kind="mergesort")
            self._sweep_azimuths.append(np.ascontiguousarray(az[az_order]))
            self._sweep_ranges.append(np.ascontiguousarray(r[r_order]))
            self._sweep_data.append(
                np.ascontiguousarray(data[np.ix_(az_order, r_order)])
            )

    def _resolve_radar_height(self) -> float:
        try:
            h_km = float(self.rl[0]["height"].values[0, 0])
            d_km = float(self._sweep_ranges[0][0])
            el = self.elev_angles[0]
            h_radar = (h_km - d_km * np.sin(np.deg2rad(el))
                       - d_km**2 / (2.0 * self.reff / 1000.0)) * 1000.0
            return max(0.0, float(h_radar))
        except (KeyError, IndexError, ValueError, TypeError):
            return 0.0

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

        ranges_sq = (radar_radius**2 + gate_radius**2 -
                     2.0 * radar_radius * gate_radius * np.cos(phi))
        ranges_sq = np.maximum(ranges_sq, 0.0)
        ranges = np.sqrt(ranges_sq)

        cos_arg = np.clip(
            (radar_radius**2 + ranges**2 - gate_radius**2) /
            (2.0 * radar_radius * np.maximum(ranges, 1e-10)),
            -1.0, 1.0,
        )
        elevation = np.rad2deg(np.arccos(cos_arg) - np.pi / 2.0)
        elevation[ranges < 1e-10] = 0.0

        azimuth = np.rad2deg(np.pi / 2.0 - np.arctan2(y, x))
        azimuth = np.mod(azimuth, 360.0)

        return azimuth, ranges, elevation

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

    def _interp_sweep_sample(self, sweep_idx, target_az, target_r, fillvalue):
        """
        在指定仰角层内做 (azimuth, range) 双线性插值。
        """
        azimuth = self._sweep_azimuths[sweep_idx]
        ranges = self._sweep_ranges[sweep_idx]
        data = self._sweep_data[sweep_idx]

        naz = len(azimuth)
        nrange = len(ranges)

        if naz < 2 or nrange < 2:
            return fillvalue

        r_km = target_r / 1000.0
        if r_km > ranges[-1] or r_km < ranges[0]:
            return fillvalue

        az_rad = np.deg2rad(target_az)
        iaz = np.searchsorted(azimuth, az_rad, side='right')

        if iaz >= naz:
            iaz = 0
            az_rad_val = az_rad - 2.0 * np.pi
        else:
            az_rad_val = az_rad

        if iaz == 0:
            az_last = azimuth[naz - 1] - 2.0 * np.pi
        else:
            az_last = azimuth[iaz - 1]

        ir = np.searchsorted(ranges, r_km, side='right')
        if ir <= 0:
            ir = 1
        elif ir >= nrange:
            ir = nrange - 1

        er0 = self._interp_linear(
            az_rad_val, az_last, azimuth[iaz],
            data[iaz - 1, ir - 1], data[iaz, ir - 1], fillvalue)
        er1 = self._interp_linear(
            az_rad_val, az_last, azimuth[iaz],
            data[iaz - 1, ir], data[iaz, ir], fillvalue)
        return self._interp_linear(
            r_km, ranges[ir - 1], ranges[ir], er0, er1, fillvalue)

    def _interp_python(self, az_deg, r_m, el_deg, beam_half):
        ne = len(self.elev_angles)
        result = np.full(len(el_deg), np.nan, dtype=np.float64)

        for i in range(len(el_deg)):
            el = el_deg[i]
            az = az_deg[i]
            r = r_m[i]

            if np.isnan(el) or np.isnan(r) or r <= 0:
                continue

            if ne == 1:
                if abs(el - self.elev_angles[0]) > beam_half:
                    continue
                result[i] = self._interp_sweep_sample(0, az, r, np.nan)
                continue

            if el < self.elev_angles[0] - beam_half:
                continue
            if el > self.elev_angles[-1] + beam_half:
                continue

            if el <= self.elev_angles[0] + beam_half:
                result[i] = self._interp_sweep_sample(0, az, r, np.nan)
            elif el >= self.elev_angles[-1] - beam_half:
                result[i] = self._interp_sweep_sample(ne - 1, az, r, np.nan)
            else:
                ie = np.searchsorted(self.elev_angles, el, side='right')
                if ie >= ne:
                    ie = ne - 1
                lower = ie - 1
                upper = ie

                val_low = self._interp_sweep_sample(lower, az, r, np.nan)
                val_up = self._interp_sweep_sample(upper, az, r, np.nan)

                if np.isnan(val_low) and np.isnan(val_up):
                    result[i] = np.nan
                elif np.isnan(val_low):
                    result[i] = val_up
                elif np.isnan(val_up):
                    result[i] = val_low
                else:
                    w = ((el - self.elev_angles[lower]) /
                         (self.elev_angles[upper] - self.elev_angles[lower]))
                    result[i] = val_low * (1.0 - w) + val_up * w

        return result

    def get_cappi_xy(self, xRange: np.ndarray, yRange: np.ndarray,
                     level_height: float, fillvalue=np.nan,
                     beam_width_deg: float = 1.0) -> Dataset:
        """
        生成笛卡尔坐标系的 CAPPI。

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

        t0 = time.time()
        az_deg, r_m, el_deg = self._cartesian_to_antenna(
            X.ravel(), Y.ravel(),
            np.full(n_total, level_height, dtype=np.float64),
            self.radar_height, self.reff)
        if self._verbose:
            print(f"[CAPPI] 坐标转换耗时: {time.time() - t0:.2f}s")

        t1 = time.time()
        beam_half = max(beam_width_deg * 0.5, 0.1)

        if HAS_NUMBA:
            from cinrad.cappi_numba import cappi_sweep_interp
            result = cappi_sweep_interp(
                X.ravel(), Y.ravel(),
                level_height, self.radar_height, self.reff,
                self._sweep_azimuths, self._sweep_ranges, self._sweep_data,
                self.elev_angles,
                fillvalue=np.nan, beam_half=beam_half,
            )
        else:
            result = self._interp_python(az_deg, r_m, el_deg, beam_half)

        if self._verbose:
            print(f"[CAPPI] 插值耗时: {time.time() - t1:.2f}s")

        result_2d = _replace_fillvalue(result.reshape(ny, nx), fillvalue)
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
