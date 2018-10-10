# -*- coding: utf-8 -*-
#Author: Du puyuan

from .utils import composite_reflectivity, echo_top, vert_integrated_liquid
from .datastruct import L2
from .grid import grid_2d, resample
from .projection import get_coordinate

import numpy as np

def _extract(Rlist):
    r_data = list()
    elev = list()
    areso = Rlist[0].a_reso if Rlist[0].a_reso else 360
    for i in Rlist:
        x, d, a = resample(i.data, i.dist, i.az, i.reso, areso)
        r_data.append(x)
        elev.append(i.elev)
    return r_data, elev

def quick_cr(Rlist):
    r_data = list()
    for i in Rlist:
        r, x, y = grid_2d(i.data, i.lon, i.lat)
        r_data.append(r)
    cr = composite_reflectivity(r_data)
    x, y = np.meshgrid(x, y)
    l2_obj = L2(cr, i.drange, 0, 1, i.code, i.name, i.time, 'cr',
                i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(x, y, np.zeros(x.shape))
    return l2_obj

def quick_et(Rlist):
    r_data, elev = _extract(Rlist)
    data = np.concatenate(r_data).reshape(len(Rlist), x.shape[0], x.shape[1])
    et = echo_top(data, d, elev, 0, 18)
    l2_obj = L2(et, i.drange, 0, 1, i.code, i.name, i.time, 'et',
                i.stp['lon'], i.stp['lat'])
    lon, lat = get_coordinate(d[0], a.T[0], 0, i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(lon, lat, np.zeros(lon.shape))
    return l2_obj

def quick_vil(Rlist):
    r_data, elev = _extract(Rlist)
    data = np.concatenate(r_data).reshape(len(Rlist), x.shape[0], x.shape[1])
    vil = vert_integrated_liquid(data, d, elev)
    l2_obj = L2(vil, i.drange, 0, 1, i.code, i.name, i.time, 'vil',
                i.stp['lon'], i.stp['lat'])
    lon, lat = get_coordinate(d[0], a.T[0], 0, i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(lon, lat, np.zeros(lon.shape))
    return l2_obj