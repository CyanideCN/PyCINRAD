# -*- coding: utf-8 -*-
# Author: Puyuan Du
from typing import Union

import numpy as np

from cinrad.constants import deg2rad, rm
from cinrad._typing import Boardcast_T, Number_T

def height(distance:Boardcast_T, elevation:Union[int ,float],
           radarheight:Number_T) -> np.ndarray:
    r'''
    Calculate height of radar beam considering atmospheric refraction.

    Parameters
    ----------
    distance: int or float or numpy.ndarray
        distance in kilometer
    elevation: int or float
        elevation angle in degree
    radarheight: int or float
        height of radar in kilometer

    Returns
    -------
    height
    '''
    return distance * np.sin(elevation * deg2rad) + distance ** 2 / (2 * rm) + radarheight / 1000

def get_coordinate(distance:Boardcast_T, azimuth:Boardcast_T,
                   elevation:Number_T, centerlon:Number_T, centerlat:Number_T,
                   h_offset=True) -> tuple:
    r'''
    Convert polar coordinates to geographic coordinates with the given radar station position.

    Parameters
    ----------
    distance: int or float or numpy.ndarray
        distance in kilometer in terms of polar coordinate
    azimuth: int or float or numpy.ndarray
        azimuth in radian in terms of polar coordinate
    elevation: int or float
        elevation angle in degree
    centerlon: int or float
        longitude of center point
    centerlat: int or float
        latitude of center point

    Returns
    -------
    actuallon: float or numpy.ndarray
        longitude value
    actuallat: float or numpy.ndarray
        latitude value
    '''
    elev = elevation if h_offset else 0
    if isinstance(azimuth, np.ndarray):
        deltav = np.cos(azimuth[:, np.newaxis]) * distance * np.cos(elev * deg2rad)
        deltah = np.sin(azimuth[:, np.newaxis]) * distance * np.cos(elev * deg2rad)
    else:
        deltav = np.cos(azimuth) * distance * np.cos(elev * deg2rad)
        deltah = np.sin(azimuth) * distance * np.cos(elev * deg2rad)
    deltalat = deltav / 111
    actuallat = deltalat + centerlat
    deltalon = deltah / (111 * np.cos(actuallat * deg2rad))
    actuallon = deltalon + centerlon
    return actuallon, actuallat