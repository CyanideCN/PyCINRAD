# -*- coding: utf-8 -*-
#Author: Du puyuan

from .constants import deg2rad, Rm1

import numpy as np

def height(distance, elevation, radarheight):
    return distance * np.sin(elevation * deg2rad) + distance ** 2 / (2 * Rm1) + radarheight / 1000

def get_coordinate(distance, azimuth, elevation, centerlon, centerlat, h_offset=True):
    r'''Convert polar coordinates to geographic coordinates with the given radar station position.'''
    elev = elevation if h_offset else 0
    deltav = np.cos(azimuth[:, np.newaxis]) * distance * np.cos(np.deg2rad(elev))
    deltah = np.sin(azimuth[:, np.newaxis]) * distance * np.cos(np.deg2rad(elev))
    deltalat = deltav / 111
    actuallat = deltalat + centerlat
    deltalon = deltah / 111
    actuallon = deltalon + centerlon
    return actuallon, actuallat