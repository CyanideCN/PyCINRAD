# -*- coding: utf-8 -*-
#Author: Du puyuan

class Raw:
    r'''Basic radar data struct'''
    def __init__(self, data, drange, elev, reso, code, name
                 , time, dtype, stlon, stlat, lon=None, lat=None
                 , height=None, a_reso=None):
        self.data = data
        self.drange = drange
        self.elev = elev
        self.reso = reso
        self.code = code
        self.name = name
        self.time = time
        self.dtype = dtype
        self.lon = lon
        self.lat = lat
        self.height = height
        self.a_reso = a_reso
        self.stp = {'lon':stlon, 'lat':stlat}
        if lon == lat == None:
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

class R(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time, stlon, stlat):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, 'r', stlon, stlat)

class V(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time, stlon, stlat, include_rf=True):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, 'v', stlon, stlat)
        self.include_rf = include_rf
        if include_rf:
            if not isinstance(data, (tuple, list)):
                raise TypeError('Expect tuple or list, get {}'.format(type(data)))

class W(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time, stlon, stlat):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, 'w', stlon, stlat)

class L2(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time, dtype, stlon, stlat):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, dtype, stlon, stlat)

class Section:
    def __init__(self, data, xcor, ycor, azimuth, drange, tstr, code, name, dtype):
        self.data = data
        self.xcor = xcor
        self.ycor = ycor
        self.az = azimuth
        self.drange = drange
        self.tstr = tstr
        self.code = code
        self.name = name
        self.dtype = dtype
        