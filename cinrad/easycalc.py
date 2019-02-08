# -*- coding: utf-8 -*-
# Author: Puyuan Du

import datetime
import time

import numpy as np
from xarray import DataArray

from cinrad.utils import echo_top, vert_integrated_liquid
from cinrad.datastruct import Radial, Grid, _Slice
from cinrad.grid import grid_2d, resample
from cinrad.projection import height, get_coordinate
from cinrad.constants import deg2rad
from cinrad.error import RadarCalculationError

__all__ = ['quick_cr', 'quick_et', 'quick_vil', 'VCS']

def _extract(r_list):
    r_data = list()
    elev = list()
    areso = r_list[0].a_reso if r_list[0].a_reso else 360
    for i in r_list:
        x, d, a = resample(i.data, i.dist, i.az, i.reso, areso)
        r_data.append(x)
        elev.append(i.elev)
    data = np.concatenate(r_data).reshape(len(r_list), r_data[0].shape[0], r_data[0].shape[1])
    return data, d, a, elev

def _nearest_ten_minute(date:datetime.datetime):
    minute = (date.minute // 10) * 10
    return datetime.datetime(date.year, date.month, date.day, date.hour, minute)

def quick_cr(r_list):
    r'''
    Calculate composite reflectivity

    Paramters
    ---------
    r_list: list of cinrad.datastruct.Radial

    Returns
    -------
    l2_obj: cinrad.datastruct.Grid
        composite reflectivity
    '''
    r_data = list()
    for i in r_list:
        r, x, y = grid_2d(i.data, i.lon, i.lat)
        r_data.append(r)
    cr = np.max(r_data, axis=0)
    x, y = np.meshgrid(x, y)
    l2_obj = Grid(np.ma.array(cr, mask=(cr <= 0)), i.drange, 1, i.code, i.name, i.scantime,
                  'CR', x, y)
    return l2_obj

def quick_et(r_list):
    r'''
    Calculate echo tops

    Paramters
    ---------
    r_list: list of cinrad.datastruct.Radial

    Returns
    -------
    l2_obj: cinrad.datastruct.Grid
        echo tops
    '''
    r_data, d, a, elev = _extract(r_list)
    i = r_list[0]
    et = echo_top(r_data, d, elev, 0)
    l2_obj = Radial(et, i.drange, 0, 1, i.code, i.name, i.scantime, 'ET',
                    i.stp['lon'], i.stp['lat'])
    lon, lat = get_coordinate(d[0], a.T[0], 0, i.stp['lon'], i.stp['lat'])
    l2_obj.add_geoc(lon, lat, np.zeros(lon.shape))
    return l2_obj

def quick_vil(r_list):
    r'''
    Calculate vertically integrated liquid

    Paramters
    ---------
    r_list: list of cinrad.datastruct.Radial

    Returns
    -------
    l2_obj: cinrad.datastruct.Grid
        vertically integrated liquid
    '''
    r_data, d, a, elev = _extract(r_list)
    i = r_list[0]
    vil = vert_integrated_liquid(r_data, d, elev)
    l2_obj = Radial(np.ma.array(vil, mask=(vil <= 0)), i.drange, 0, 1, i.code, i.name, i.scantime,
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
            r_data.append(r)
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
        sl = _Slice(r, x, h, self.rl[0].scantime, self.rl[0].code, self.rl[0].name, 'VCS', stp_s=stp_s,
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

        Returns
        -------
        sl: cinrad.datastruct._Slice
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

class RadarMosaic(object):
    r'''Untested'''
    def __init__(self, data):
        self.data_list = list()
        self.add_data(data)

    def _check_time(self):
        d_list = [_nearest_ten_minute(i.scantime) for i in self.data_list]
        d_list_stamp = [time.mktime(i.timetuple()) for i in d_list]
        if np.average(d_list_stamp) != d_list_stamp[0]:
            raise RadarCalculationError('Input radar data have inconsistent time')

    def add_data(self, data):
        if isinstance(data, Radial):
            self.data_list.append(data)
        elif isinstance(data, (list, tuple)):
            for i in data:
                self.data_list.append(i)
        self._check_time()

    def gen_longitude(self, extent=None, points=1000):
        if extent:
            return np.linspace(extent[0], extent[1], points)
        else:
            left_coor = np.min([np.min(i.lon) for i in self.data_list])# - 0.5
            right_coor = np.max([np.max(i.lon) for i in self.data_list])# + 0.5
            return np.linspace(left_coor, right_coor, points)

    def gen_latitude(self, extent=None, points=1000):
        if extent:
            return np.linspace(extent[0], extent[1], points)
        else:
            bottom_coor = np.min([np.min(i.lat) for i in self.data_list])# - 0.5
            top_coor = np.max([np.max(i.lat) for i in self.data_list])# + 0.5
            return np.linspace(bottom_coor, top_coor, points)

    def merge(self):
        lon = self.gen_longitude()
        lat = self.gen_latitude()
        data_tmp = list()
        for i in self.data_list:
            data, x, y = grid_2d(i.data, i.lon, i.lat, x_out=lon, y_out=lat)
            data_tmp.append(np.ma.array(data, mask=(data == 0)))
        out_data = np.ma.average(data_tmp, axis=0)
        return x, y, out_data
