# -*- coding: utf-8 -*-
#Author: Du puyuan

from .constants import deg2rad
from .projection import height

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

def vert_integrated_liquid(ref, distance, elev, threshold=18.):
    r'''Calculate vertically integrated liquid (VIL) in one full scan
    Parameters:
    ref: reflectivity data  numpy.ndarray dim=3 (elevation angle, distance, azimuth)
    distance: distance from radar site  numpy.ndarray dim=2 (distance, azimuth)
    elev: elevation angles in degree  numpy.ndarray or list dim=1
    threshold: float
    '''
    const = 3.44e-6
    v_beam_width = 0.99 * deg2rad
    elev = np.array(elev) * deg2rad
    VIL = list()
    xshape, yshape = ref[0].shape
    for i in range(xshape):
        for j in range(yshape):
            vert_r = list()
            dist = distance[i][j] * 1000
            for k in range(0, 9):
                #index from lowest angle
                r_pt = ref[k][i][j]
                vert_r.append(r_pt)
            r_ = np.clip(vert_r, None, 55) #reduce the influence of hails
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

def echo_top(ref, distance, elev, radarheight, threshold):
    r'''Calculate height of echo tops (ET) in one full scan
    ref: reflectivity data  numpy.ndarray dim=3 (elevation angle, distance, azimuth)
    distance: distance from radar site  numpy.ndarray dim=2 (distance, azimuth)
    elev: elevation angles in degree  numpy.ndarray or list dim=1
    radarheight: height of radar  float
    drange: range of data to be calculated  float or int
    threshold: float
    '''
    et = list()
    r = np.ma.array(ref, mask=(ref > threshold))
    xshape, yshape = r[0].shape
    h_ = list()
    for i in elev:
        h = height(distance, i, radarheight)
        h_.append(h)
    hght = np.concatenate(h_).reshape(r.shape)
    h_mask = hght * r.mask
    for i in range(xshape):
        for j in range(yshape):
            vert_h = list()
            vert_r = list()
            vert_h_ = list()
            for k in range(1, 10):
                #index from highest angle
                h_pt = h_mask[-1 * k][i][j]
                r_pt = ref[-1 * k][i][j]
                h_pt_ = hght[-1 * k][i][j]
                vert_h.append(h_pt)
                vert_r.append(r_pt)
                vert_h_.append(h_pt_)
            vertical = np.array(vert_h)
            position = np.where(vertical > 0)[0]
            try:
                pos = position[0]
            except IndexError:#empty array
                et.append(0)
                continue
            if pos == 0:
                height_ = vertical[pos]
                et.append(height_)
            else:
                try:
                    elev[pos - 1]
                except IndexError:
                    et.append(vertical[pos])
                    continue
                z1 = vert_r[pos]
                z2 = vert_r[pos - 1]
                h1 = vertical[pos]
                h2 = vert_h_[pos - 1]
                w1 = (z1 - threshold) / (z1 - z2)
                w2 = 1 - w1
                et.append(w1 * h2 + w2 * h1)
    return np.array(et).reshape(xshape, yshape)