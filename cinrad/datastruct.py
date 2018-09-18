# -*- coding: utf-8 -*-
#Author: Du puyuan

class Raw:
    def __init__(self, data, drange, elev, reso, code, name
                 , time, dtype, lon=None, lat=None, height=None):
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
        if lon == lat == None:
            self.geoflag = False
        else:
            self.geoflag = True

    def __repr__(self):
        repr_s = ('Datatype: {}\nStation name:{}\nScan time:{}\nElevation angle:' 
        + '{}\nRange:{}')
        return repr_s.format(
            self.dtype.upper(), self.name, self.time, self.elev, self.drange)

    def add_geoc(self, lon, lat, height):
        if not lon.shape == lat.shape == height.shape:
            raise ValueError('Coordinate sizes are incompatible')
        self.lon = lon
        self.lat = lat
        self.height = height
        self.geoflag = True

class R(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, 'r')

class V(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time, include_rf=True):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, 'v')
        self.include_rf = include_rf
        if include_rf:
            if not isinstance(data, (tuple, list)):
                raise TypeError('Expect tuple or list, get {}'.format(type(data)))

class L2(Raw):
    def __init__(self, data, drange, elev, reso, code, name, time, dtype):
        Raw.__init__(self, data, drange, elev, reso, code, name, time, dtype)
