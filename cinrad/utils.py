# -*- coding: utf-8 -*-
# Author: Puyuan Du

from .constants import deg2rad
from .projection import height

import numpy as np

def mask_outside(data, drange):
    r'''
    Mask data outside obervation range
    '''
    xdim = data.shape[0]
    xcoor = np.linspace(-1 * drange, drange, xdim)
    x, y = np.meshgrid(xcoor, xcoor)
    dist = np.sqrt(np.abs(x ** 2 + y ** 2))
    return np.ma.array(data, mask=(dist > drange))

def composite_reflectivity(ref, drange=230):
    r'''
    Find max ref value in single coordinate

    Parameters
    ----------
    ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
        reflectivity data
    drange: float or int
        data range

    Returns
    -------
    data: numpy.ndarray
        composite reflectivity data
    '''
    r_max = np.max(ref, axis=0)
    data = mask_outside(r_max, drange)
    return data

def vert_integrated_liquid(ref, distance, elev, threshold=18.):
    r'''
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
    '''
    const = 3.44e-6
    v_beam_width = 0.99 * deg2rad
    elev = np.array(elev) * deg2rad
    xshape, yshape = ref[0].shape
    distance *= 1000
    hi_arr = distance * np.sin(v_beam_width / 2)
    VIL = np.zeros((xshape, yshape))
    for i in range(xshape):
        for j in range(yshape):
            vert_r = ref[:, i, j]
            dist = distance[i][j]
            r_ = np.clip(vert_r, None, 55) #reduce the influence of hails
            vertical = 10 ** (r_ / 10)
            position = np.where(r_ > threshold)[0]
            if position.shape[0] == 0:
                VIL[i][j] = 0
                continue
            pos_s = position[0]
            pos_e = position[-1]
            m1 = 0
            hi = hi_arr[i][j]
            for l in position[:-1].astype(int):
                ht = dist * (np.sin(elev[l + 1]) - np.sin(elev[l]))
                factor = ((vertical[l] + vertical[l + 1]) / 2) ** (4 / 7)
                m1 += const * factor * ht
            mb = const * vertical[pos_s] ** (4 / 7) * hi
            mt = const * vertical[pos_e] ** (4 / 7) * hi
            VIL[i][j] = m1 + mb + mt
    return VIL

def echo_top(ref, distance, elev, radarheight, threshold=18.):
    r'''
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
    '''
    r = np.ma.array(ref, mask=(ref > threshold))
    xshape, yshape = r[0].shape
    et = np.zeros((xshape, yshape))
    h_ = list()
    for i in elev:
        h = height(distance, i, radarheight)
        h_.append(h)
    hght = np.concatenate(h_).reshape(r.shape)
    for i in range(xshape):
        for j in range(yshape):
            vert_h = hght[:, i, j]
            vert_r = ref[:, i, j]
            if vert_r.max() < threshold: # Vertical points don't satisfy threshold
                et[i][j] = 0
                continue
            elif vert_r[-1] >= threshold: # Point in highest scan exceeds threshold
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