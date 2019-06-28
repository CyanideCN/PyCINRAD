# -*- coding: utf-8 -*-
# Author: Puyuan Du
from datetime import datetime
from typing import Union, Optional, Any
from copy import deepcopy as dc

from numpy import ndarray

class Radial(object):
    r'''Structure for data arranged by radials'''

    __slots__ = ['data', 'drange', 'elev', 'reso', 'code', 'name', 'scantime', 'dtype', 'include_rf',
                    'lon', 'lat', 'height', 'a_reso', 'stp', 'geoflag', 'dist', 'az', 'scan_info']

    def __init__(self, data:ndarray, drange:Union[float, int], elev:float, reso:float, code:str, name:str,
                 scantime:datetime, dtype:str, stlon:float, stlat:float, lon:Optional[ndarray]=None,
                 lat:Optional[ndarray]=None, height:Optional[ndarray]=None, a_reso:Optional[int]=None, **scan_info):
        r'''
        Parameters
        ----------
        data: np.ndarray
            wrapped data
        drange: float
            radius of this data
        elev: float
            elevation angle of this data
        reso: float
            radial resolution of this data
        code: str
            code for this radar
        name: str
            name for this radar
        scantime: str
            scan time for this radar
        dtype: str
            product type
        stlon: float
            radar longitude
        stlat: float
            radar latitude
        lon: np.ndarray / bool
            longitude array for wrapped data
        lat: np.ndarray / bool
            latitude array for wrapped data
        height: np.ndarray / bool
            height array for wrapped data
        a_reso: int
            radial resolution of this data
        scan_info: dict
            scan parameters of radar
        '''
        self.data = data
        self.drange = drange
        self.elev = elev
        self.reso = reso
        self.code = code
        self.name = name
        self.scantime = scantime
        self.dtype = dtype
        self.scan_info = scan_info
        if dtype == 'VEL':
            if len(data) == 2:
                self.include_rf = True
            else:
                self.include_rf = False
        self.lon = lon
        self.lat = lat
        self.height = height
        self.a_reso = a_reso
        self.stp = {'lon':stlon, 'lat':stlat}
        nonetype = type(None)
        if isinstance(lon, nonetype) and isinstance(lat, nonetype):
            self.geoflag = False
        else:
            self.geoflag = True

    def __repr__(self):
        repr_s = ('Datatype: {}\nStation name: {}\nScan time: {}\nElevation angle: '
        + '{}\nRange: {}')
        return repr_s.format(
            self.dtype.upper(), self.name, self.scantime, self.elev, self.drange)

    def add_geoc(self, lon:ndarray, lat:ndarray, height:ndarray):
        if not lon.shape == lat.shape == height.shape:
            raise ValueError('Coordinate sizes are incompatible')
        self.lon = lon
        self.lat = lat
        self.height = height
        self.geoflag = True

    def add_polarc(self, distance:ndarray, azimuth:ndarray):
        self.dist = distance
        self.az = azimuth

    def __deepcopy__(self, memo:Any):
        r'''Used if copy.deepcopy is called'''
        r = Radial(dc(self.data), dc(self.drange), dc(self.elev), dc(self.reso), dc(self.code),
                   dc(self.name), dc(self.scantime), dc(self.dtype), dc(self.stp['lon']), dc(self.stp['lat']),
                   dc(self.scan_info))
        if self.geoflag:
            r.add_geoc(dc(self.lon), dc(self.lat), dc(self.height))
        if hasattr(self, 'dist'):
            r.add_polarc(dc(self.dist), dc(self.az))
        return r

class Slice_(object):
    r'''Structure for slice data'''

    __slots__ = ['data', 'xcor', 'ycor', 'scantime', 'dtype', 'code', 'name', 'geoinfo']

    def __init__(self, data:ndarray, xcor:ndarray, ycor:ndarray, scantime:datetime, code:str,
                 name:str, dtype:str, **geoinfo):
        self.data = data
        self.xcor = xcor
        self.ycor = ycor
        self.geoinfo = geoinfo
        self.scantime = scantime
        self.code = code
        self.name = name
        self.dtype = dtype

class Grid(object):
    r'''Structure for processed grid data'''

    __slots__ = ['data', 'drange', 'reso', 'code', 'name', 'scantime', 'dtype', 'lon', 'lat', 'geoflag', 'elev',
                 'scan_info']

    def __init__(self, data:ndarray, drange:Union[float, int], reso:float, code:str, name:str,
                 scantime:datetime, dtype:str, lon:ndarray, lat:ndarray, **scan_info):
        self.data = data
        self.drange = drange
        self.reso = reso
        self.code = code
        self.name = name
        self.scantime = scantime
        self.dtype = dtype
        self.lon = lon
        self.lat = lat
        self.geoflag = True
        self.elev = 0
        self.scan_info = scan_info

    def __repr__(self):
        repr_s = ('Datatype: {}\nStation name: {}\nScan time: {}\n')
        return repr_s.format(
            self.dtype.upper(), self.name, self.scantime)