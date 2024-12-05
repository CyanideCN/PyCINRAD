# -*- coding: utf-8 -*-
# Author: Puyuan Du

from typing import Union, Any

import numpy as np
import math

from cinrad.constants import deg2rad, vil_const
from cinrad.projection import height
from cinrad._typing import Array_T, Number_T

def r2z(r: np.ndarray) -> np.ndarray:
    return 10 ** (r / 10)

def vert_integrated_liquid_py(
    ref: np.ndarray,
    distance: np.ndarray,
    elev: Array_T,
    beam_width: float = 0.99,
    threshold: Union[float, int] = 18.0,
    density: bool = False,
) -> np.ndarray:
    r"""
    Calculate vertically integrated liquid (VIL) in one full scan

    Parameters
    ----------
    ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
        reflectivity data
    distance: numpy.ndarray dim=2 (distance, azimuth)
        distance from radar site
    elev: numpy.ndarray or list dim=1
        elevation angles in degree
    threshold: float
        minimum reflectivity value to take into calculation

    Returns
    -------
    data: numpy.ndarray
        vertically integrated liquid data
    """
    if density:
        raise NotImplementedError("VIL density calculation is not implemented")
    v_beam_width = beam_width * deg2rad
    elev = np.array(elev) * deg2rad
    xshape, yshape = ref[0].shape
    distance *= 1000
    hi_arr = distance * np.sin(v_beam_width / 2)
    vil = _vil_iter(xshape, yshape, ref, distance, elev, hi_arr, threshold)
    return vil

def _vil_iter(
    xshape: int,
    yshape: int,
    ref: np.ndarray,
    distance: np.ndarray,
    elev: Array_T,
    hi_arr: np.ndarray,
    threshold: Number_T,
) -> np.ndarray:
    # r = np.clip(ref, None, 55) #reduce the influence of hails
    r = ref
    z = r2z(r)
    VIL = np.zeros((xshape, yshape))
    for i in range(xshape):
        for j in range(yshape):
            vert_r = r[:, i, j]
            vert_z = z[:, i, j]
            dist = distance[i][j]
            position = np.where(vert_r > threshold)[0]
            if position.shape[0] == 0:
                continue
            pos_s = position[0]
            pos_e = position[-1]
            m1 = 0
            hi = hi_arr[i][j]
            for l in range(pos_e):
                ht = dist * (np.sin(elev[l + 1]) - np.sin(elev[l]))
                factor = ((vert_z[l] + vert_z[l + 1]) / 2) ** (4 / 7)
                m1 += vil_const * factor * ht
            mb = vil_const * vert_z[pos_s] ** (4 / 7) * hi
            mt = vil_const * vert_z[pos_e] ** (4 / 7) * hi
            VIL[i][j] = m1 + mb + mt
    return VIL

def echo_top_py(
    ref: np.ndarray,
    distance: np.ndarray,
    elev: Array_T,
    radarheight: Number_T,
    threshold: Number_T = 18.0,
) -> np.ndarray:
    r"""
    Calculate height of echo tops (ET) in one full scan

    Parameters
    ----------
    ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
        reflectivity data
    distance: numpy.ndarray dim=2 (distance, azimuth)
        distance from radar site
    elev: numpy.ndarray or list dim=1
        elevation angles in degree
    radarheight: int or float
        height of radar
    drange: float or int
        range of data to be calculated
    threshold: float
        minimum value of reflectivity to be taken into calculation

    Returns
    -------
    data: numpy.ndarray
        echo tops data
    """
    xshape, yshape = ref[0].shape
    et = np.zeros((xshape, yshape))
    h_ = list()
    for i in elev:
        h = height(distance, i, radarheight)
        h_.append(h)
    hght = np.concatenate(h_).reshape(ref.shape)
    for i in range(xshape):
        for j in range(yshape):
            vert_h = hght[:, i, j]
            vert_r = ref[:, i, j]
            if vert_r.max() < threshold:  # Vertical points don't satisfy threshold
                et[i][j] = 0
                continue
            elif vert_r[-1] >= threshold:  # Point in highest scan exceeds threshold
                et[i][j] = vert_h[-1]
                continue
            else:
                position = np.where(vert_r >= threshold)[0]
                if position[-1] == 0:
                    et[i][j] = vert_h[0]
                    continue
                else:
                    pos = position[-1]
                    z1 = vert_r[pos]
                    z2 = vert_r[pos + 1]
                    h1 = vert_h[pos]
                    h2 = vert_h[pos + 1]
                    w1 = (z1 - threshold) / (z1 - z2)
                    w2 = 1 - w1
                    et[i][j] = w1 * h2 + w2 * h1
    return et

def calculate_distance_py(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    This function uses the Haversine formula which takes into account the spherical shape of the Earth
    approximated as a sphere with an average radius.

    Args:
        lon1 (float): Longitude of the first point.
        lat1 (float): Latitude of the first point.
        lon2 (float): Longitude of the second point.
        lat2 (float): Latitude of the second point.

    Returns:
        float: The distance between the two points in meters.
    """
    R = 6371000  # Earth's average radius in meters
    # Convert latitude and longitude values from degrees to radians to be used in trigonometric calculations
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    # Calculate intermediate value 'a' as per the Haversine formula
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def calculate_azimuth_py(lon1, lat1, lon2, lat2):
    """
    Calculate the azimuth (bearing) from the first point to the second point on the Earth's surface.

    The azimuth is the angle measured clockwise from the north direction.

    Args:
        lon1 (float): Longitude of the starting point.
        lat1 (float): Latitude of the starting point.
        lon2 (float): Longitude of the destination point.
        lat2 (float): Latitude of the destination point.

    Returns:
        float: The azimuth in radians.
    """
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    # Calculate the differences in x and y directions on the spherical coordinate system
    dx = math.cos(lat2_rad) * math.sin(lon2_rad - lon1_rad)
    dy = (math.cos(lat1_rad) * math.sin(lat2_rad) -
          math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad))
    azimuth = math.atan2(dx, dy)
    azimuth = math.degrees(azimuth)
    # Adjust the azimuth to be in the range of 0 to 360 degrees
    azimuth %= 360  
    return math.radians(azimuth)

def geo_to_polar_py(lon, lat, center_lon, center_lat):
    """
    Convert geographical coordinates (longitude and latitude) to polar coordinates (azimuth and distance)
    relative to a given center point.

    Args:
        lon (float): Longitude of the point to be converted.
        lat (float): Latitude of the point to be converted.
        center_lon (float): Longitude of the center reference point.
        center_lat (float): Latitude of the center reference point.

    Returns:
        tuple: A tuple containing the azimuth (in radians) and the distance (in kilometers).
    """
    azimuth = calculate_azimuth_py(center_lon, center_lat, lon, lat)
    distance = calculate_distance_py(center_lon, center_lat, lon, lat) / 1e3  # Convert distance to kilometers
    return azimuth, distance

try:
    from cinrad._utils import *
except ImportError:
    # When the C-extension doesn't exist, define the functions in Python.
    echo_top = echo_top_py
    vert_integrated_liquid = vert_integrated_liquid_py
    calculate_distance = calculate_distance_py
    calculate_azimuth = calculate_azimuth_py
    geo_to_polar = geo_to_polar_py