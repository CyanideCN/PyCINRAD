# -*- coding: utf-8 -*-
#Author: Du puyuan

from .constants import deg2rad
import calc

import numpy as np

def composite_reflectivity(ref, drange=230):
    r'''Find max ref value in single coordinate and mask data outside obervation range
    Parameters:
    ref: reflectivity data  numpy.ndarray dim=3 (elevation angle, distance, azimuth)
    drange: float or int
    '''
    r_max = np.max(ref, axis=0)
    xdim = r_max.shape[0]
    xcoor = np.linspace(-1 * drange, drange, xdim)
    x, y = np.meshgrid(xcoor, xcoor)
    dist = np.sqrt(np.abs(x ** 2 + y ** 2))
    return np.ma.array(r_max, mask=(dist > drange))

def vert_integrated_liquid(ref, distance, elev, drange, threshold=18.):
    r'''Calculate vertically integrated liquid (VIL) in one full scan
    Parameters:
    ref: reflectivity data  numpy.ndarray dim=3 (elevation angle, distance, azimuth)
    distance: distance from radar site  numpy.ndarray dim=2 (distance, azimuth)
    elev: elevation angles in degree  numpy.ndarray or list dim=1
    drange: range of data to be calculated  float or int
    threshold: float
    '''
    const = 3.44e-6
    v_beam_width = 0.99 * deg2rad
    VIL = list()
    xshape, yshape = ref[0].shape
    for i in range(xshape):
        for j in range(yshape):
            vert_r = list()
            dist = distance[i][j] * 1000
            for k in range(0, 9):
                #index from lowest angle
                r_pt = r[k][i][j]
                vert_r.append(r_pt)
            r_ = np.array(vert_r)
            vertical = 10 ** (r_ / 10)
            position = np.where(r_ > threshold)[0]
            try:
                pos_s = position[0]
                pos_e = position[-1]
            except IndexError:
                VIL.append(0)
                continue
            m1 = 0
            hi = dist * np.sin(v_beam_width / 2)
            for l in position[:-1].astype(int):
                ht = dist * (np.sin(elev[l + 1]) - np.sin(elev[l]))
                factor = ((vertical[l] + vertical[l + 1]) / 2) ** (4 / 7)
                m1 += const * factor * ht
            mb = const * vertical[pos_s] ** (4 / 7) * hi
            mt = const * vertical[pos_e] ** (4 / 7) * hi
            VIL.append(m1 + mb + mt)
    return np.array(VIL).reshape(xshape, yshape)

def echo_top(ref, distance, elev, siteheight, drange, threshold=18.):
    r'''Calculate height of echo tops (ET) in one full scan
    ref: reflectivity data  numpy.ndarray dim=3 (elevation angle, distance, azimuth)
    distance: distance from radar site  numpy.ndarray dim=2 (distance, azimuth)
    elev: elevation angles in degree  numpy.ndarray or list dim=1
    siteheight: height of radar  float
    drange: range of data to be calculated  float or int
    threshold: float
    '''
    data = [ref, distance]
    et = calc.echo_top(data, elev, siteheight, threshold)
    return et