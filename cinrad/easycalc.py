# -*- coding: utf-8 -*-
#Author: Du puyuan

from .utils import composite_reflectivity, echo_top, vert_integrated_liquid
from .datastruct import L2
from .grid import grid_2d, resample

import numpy as np

def quick_cr(Rlist):
    r_data = list()
    for i in Rlist:
        r, x, y = grid_2d(i.data, i.lon, i.lat)
        r_data.append(r)
    cr = composite_reflectivity(r_data)
    x, y = np.meshgrid(x, y)
    l2_obj = L2(cr, i.drange, 0, 1, i.code, i.name, i.time, 'cr')
    l2_obj.add_geoc(x, y, np.zeros(x.shape))
    return l2_obj

def quick_et(Rlist):
    r_data = list()
    elev = list()
    for i in Rlist:
        x, d, a = resample(i.data, i.dist, i.az, i.reso)
        r_data.append(x)
        elev.append(i.elev)
    data = np.concatenate(r_data).reshape(9, 361, 230)
    et = echo_top(data, d, elev)
    l2_obj = L2(et, i.drange, 0, 1, i.code, i.name, i.time, 'et')
    l2_obj.add_geoc(i.lon, i.lat, np.zeros(i.lon.shape))
    return l2_obj

def quick_vil(Rlist):
    r_data = list()
    elev = list()
    for i in Rlist:
        x, d, a = resample(i.data, i.dist, i.az, i.reso)
        r_data.append(x)
        elev.append(i.elev)
    data = np.concatenate(r_data).reshape(9, 361, 230)
    vil = vert_integrated_liquid(data, d, elev)
    l2_obj = L2(vil, i.drange, 0, 1, i.code, i.name, i.time, 'vil')
    l2_obj.add_geoc(i.lon, i.lat, np.zeros(i.lon.shape))
    return l2_obj