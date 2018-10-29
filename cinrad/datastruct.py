# -*- coding: utf-8 -*-
# Author: Puyuan Du

class Radial:
    r'''Structure for data arranged by radials'''
    def __init__(self, data, drange, elev, reso, code, name,
                 time, dtype, stlon, stlat, lon=None, lat=None,
                 height=None, a_reso=None):
        self.data = data
        self.drange = drange
        self.elev = elev
        self.reso = reso
        self.code = code
        self.name = name
        self.time = time
        self.dtype = dtype
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
            self.dtype.upper(), self.name, self.time, self.elev, self.drange)

    def add_geoc(self, lon, lat, height):
        if not lon.shape == lat.shape == height.shape:
            raise ValueError('Coordinate sizes are incompatible')
        self.lon = lon
        self.lat = lat
        self.height = height
        self.geoflag = True

    def add_polarc(self, distance, azimuth):
        self.dist = distance
        self.az = azimuth

class _Slice:
    r'''Structure for slice data'''
    def __init__(self, data, xcor, ycor, tstr, code, name, dtype, **geoinfo):
        self.data = data
        self.xcor = xcor
        self.ycor = ycor
        self.geoinfo = geoinfo
        self.tstr = tstr
        self.code = code
        self.name = name
        self.dtype = dtype

class Grid:
    r'''Structure for processed grid data'''
    def __init__(self, data, drange, reso, code, name
                 , time, dtype, lon, lat):
        self.data = data
        self.drange = drange
        self.reso = reso
        self.code = code
        self.name = name
        self.time = time
        self.dtype = dtype
        self.lon = lon
        self.lat = lat
        self.geoflag = True
        self.elev = 0

    def __repr__(self):
        repr_s = ('Datatype: {}\nStation name: {}\nScan time: {}\n')
        return repr_s.format(
            self.dtype.upper(), self.name, self.time)