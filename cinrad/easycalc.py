# -*- coding: utf-8 -*-
# Author: Puyuan Du

from .utils import composite_reflectivity, echo_top, vert_integrated_liquid, mask_outside
from .datastruct import Radial, Grid, _Slice
from .grid import grid_2d, resample
from .projection import height, get_coordinate
from .constants import deg2rad
from .error import RadarCalculationError

import numpy as np
from xarray import DataArray

__all__ = ['quick_cr', 'quick_et', 'quick_vil', 'VCS']

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
    l2_obj = Grid(np.ma.array(cr, mask=(cr <= 0)), i.drange, 1, i.code, i.name, i.time,
                  'CR', x, y)
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
    l2_obj = Radial(np.ma.array(vil, mask=(vil <= 0)), i.drange, 0, 1, i.code, i.name, i.time,
                    'VIL', i.stp['lon'], i.stp['lat'])
    lon, lat = get_coordinate(d[0], a.T[0], 0, i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(lon, lat, np.zeros(lon.shape))
    return l2_obj

class VCS:
    r'''Class performing vertical cross-section calculation'''
    def __init__(self, r_list):
        self.rl = r_list
        self.el = [i.elev for i in r_list]
        self.x, self.y, self.h, self.r = self._geocoor()

    def _geocoor(self):
        r_data = list()
        x_data = list()
        y_data = list()
        h_data = list()
        for i in self.rl:
            r, x, y = grid_2d(i.data, i.lon, i.lat)
            r_data.append(mask_outside(r, 230 * np.cos(self.el[0] * deg2rad)))
            x_data.append(x)
            y_data.append(y)
        for radial, elev in zip(self.rl, self.el):
            hgh = height(radial.dist, elev, 0)
            hgh_radial = np.asarray(hgh.tolist() * radial.data.shape[0]).reshape(radial.data.shape)
            hgh_grid, x, y = grid_2d(hgh_radial, radial.lon, radial.lat)
            h_data.append(hgh_grid)
        return x_data, y_data, h_data, r_data

    def _get_section(self, stp, enp, spacing):
        r_sec = list()
        h_sec = list()
        for x, y, h, r in zip(self.x, self.y, self.h, self.r):
            d_x = DataArray(r, [('lat', y), ('lon', x)])
            d_h = DataArray(h, [('lat', y), ('lon', x)])
            x_new = DataArray(np.linspace(stp[0], enp[0], spacing), dims='z')
            y_new = DataArray(np.linspace(stp[1], enp[1], spacing), dims='z')
            r_section = d_x.interp(lon=x_new, lat=y_new)
            h_section = d_h.interp(lon=x_new, lat=y_new)
            r_sec.append(r_section)
            h_sec.append(h_section)
        r = np.asarray(r_sec)
        h = np.asarray(h_sec)
        r[np.isnan(r)] = 0
        x = np.linspace(0, 1, spacing) * np.ones(r.shape[0])[:, np.newaxis]
        stp_s = '{}N, {}E'.format(stp[1], stp[0])
        enp_s = '{}N, {}E'.format(enp[1], enp[0])
        sl = _Slice(r, x, h, self.rl[0].time, self.rl[0].code, self.rl[0].name, 'VCS', stp_s=stp_s,
                    enp_s=enp_s, stp=stp, enp=enp)
        return sl

    def get_section(self, start_polar=None, end_polar=None, start_cart=None, end_cart=None,
                    spacing=100):
        r'''
        Get cross-section data from input points

        Parameters
        ----------
        start_polar: list or tuple
            polar coordinates of start point i.e.(distance, azimuth)
        end_polar: list or tuple
            polar coordinates of end point i.e.(distance, azimuth)
        start_cart: list or tuple
            geographic coordinates of start point i.e.(longitude, latitude)
        end_cart: list or tuple
            geographic coordinates of end point i.e.(longitude, latitude)
        '''
        if start_polar and end_polar:
            stlat = self.rl[0].stp['lat']
            stlon = self.rl[0].stp['lon']
            stp = np.round_(get_coordinate(start_polar[0], start_polar[1] * deg2rad, 0, stlon, stlat), 2)
            enp = np.round_(get_coordinate(end_polar[0], end_polar[1] * deg2rad, 0, stlon, stlat), 2)
        elif start_cart and end_cart:
            stp = start_cart
            enp = end_cart
        else:
            raise RadarCalculationError('Invalid input')
        return self._get_section(stp, enp, spacing)