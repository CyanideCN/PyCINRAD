# -*- coding: utf-8 -*-
# Author: Puyuan Du

from typing import Tuple, Optional

import numpy as np

try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import KDTree

from cinrad.constants import deg2rad
from cinrad._typing import Number_T


class KDResampler(object):
    def __init__(
        self, data: np.ndarray, x: np.ndarray, y: np.ndarray, roi: Number_T = 0.02
    ):
        x_ravel = x.ravel()
        y_ravel = y.ravel()
        self.tree = KDTree(np.dstack((x_ravel, y_ravel))[0])
        self.data = data
        self.roi = roi

    def map_data(self, x_out: np.ndarray, y_out: np.ndarray) -> np.ma.MaskedArray:
        out_coords = np.dstack((x_out.ravel(), y_out.ravel()))[0]
        _, indices = self.tree.query(out_coords, distance_upper_bound=self.roi)
        invalid_mask = indices == self.tree.n
        indices[invalid_mask] = 0
        data = np.ma.array(self.data.ravel()[indices], mask=invalid_mask)
        return data.reshape(x_out.shape)


def resample(
    data: np.ndarray,
    distance: np.ndarray,
    azimuth: np.ndarray,
    d_reso: Number_T,
    a_reso: int,
) -> tuple:
    r"""
    Resample radar radial data which have different number of radials
    in one scan into that of 360 radials

    Parameters
    ----------
    data: numpy.ndarray
        radar radial data
    distance: numpy.ndarray
        original distance
    azimuth: numpy.ndarray
        original azimuth

    Returns
    -------
    r: numpy.ndarray
        resampled radial data
    dist: numpy.ndarray
        resampled distance
    theta: numpy.ndarray
        resampled azimuth
    """
    # Target grid
    Rrange = np.arange(d_reso, distance.max() + d_reso, d_reso)
    Trange = np.linspace(0, 360, a_reso + 1) * deg2rad
    dist, theta = np.meshgrid(Rrange, Trange)
    # Original grid
    d, t = np.meshgrid(distance, azimuth)
    kds = KDResampler(data, d, t, 1)
    r = kds.map_data(dist, theta)
    return r, dist, theta


def grid_2d(
    data: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    x_out: Optional[np.ndarray] = None,
    y_out: Optional[np.ndarray] = None,
    resolution: tuple = (1000, 1000),
) -> tuple:
    r"""
    Interpolate data in polar coordinates into geographic coordinates

    Parameters
    ----------
    data: numpy.ndarray
        original radial data
    x: numpy.ndarray
        original longitude data arranged in radials
    y: numpy.ndarray
        original latitude data arranged in radials
    resolution: tuple
        the size of output

    Returns:
    r: numpy.ndarray
        interpolated data in grid
    x_cor: numpy.ndarray
        interpolated longitude in grid
    y_cor: numpy.ndarray
        interpolated latitude in grid
    """
    r_x, r_y = resolution
    if isinstance(x_out, type(None)):
        x_out = np.linspace(x.min(), x.max(), r_x)
    if isinstance(y_out, type(None)):
        y_out = np.linspace(y.min(), y.max(), r_y)
    t_x, t_y = np.meshgrid(x_out, y_out)
    kds = KDResampler(data, x, y)
    # TODO: Rewrite the logic for conversion between np.ma.masked and np.nan
    result = kds.map_data(t_x, t_y).data
    return result, x_out, y_out
