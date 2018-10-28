# -*- coding: utf-8 -*-
# Author: Puyuan Du

from .utils import composite_reflectivity, echo_top, vert_integrated_liquid
from .datastruct import Radial, Grid
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
    return r_data, d, a, elev

def quick_cr(Rlist):
    r'''
    Calculate composite reflectivity

    Paramters
    ---------
    Rlist: list of cinrad.datastruct.Radial

    Returns
    -------
    l2_obj: cinrad.datastruct.Grid
        composite reflectivity
    '''
    r_data = list()
    for i in Rlist:
        r, x, y = grid_2d(i.data, i.lon, i.lat)
        r_data.append(r)
    cr = composite_reflectivity(r_data)
    x, y = np.meshgrid(x, y)
    l2_obj = Grid(np.ma.array(cr, mask=(cr <= 0)), i.drange, 1, i.code, i.name, i.time
                , 'CR', x, y)
    return l2_obj

def quick_et(Rlist):
    r'''
    Calculate echo tops

    Paramters
    ---------
    Rlist: list of cinrad.datastruct.Radial

    Returns
    -------
    l2_obj: cinrad.datastruct.Grid
        echo tops
    '''
    r_data, d, a, elev = _extract(Rlist)
    i = Rlist[0]
    data = np.concatenate(r_data).reshape(len(Rlist), r_data[0].shape[0], r_data[0].shape[1])
    et = echo_top(data, d, elev, 0)
    l2_obj = Radial(et, i.drange, 0, 1, i.code, i.name, i.time, 'ET',
                i.stp['lon'], i.stp['lat'])
    lon, lat = get_coordinate(d[0], a.T[0], 0, i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(lon, lat, np.zeros(lon.shape))
    return l2_obj

def quick_vil(Rlist):
    r'''
    Calculate vertically integrated liquid

    Paramters
    ---------
    Rlist: list of cinrad.datastruct.Radial

    Returns
    -------
    l2_obj: cinrad.datastruct.Grid
        vertically integrated liquid
    '''
    r_data, d, a, elev = _extract(Rlist)
    i = Rlist[0]
    data = np.concatenate(r_data).reshape(len(Rlist), r_data[0].shape[0], r_data[0].shape[1])
    vil = vert_integrated_liquid(data, d, elev)
    l2_obj = Radial(np.ma.array(vil, mask=(vil <= 0)), i.drange, 0, 1, i.code, i.name, i.time
                , 'VIL', i.stp['lon'], i.stp['lat'])
    lon, lat = get_coordinate(d[0], a.T[0], 0, i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(lon, lat, np.zeros(lon.shape))
    return l2_obj