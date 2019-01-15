# -*- coding: utf-8 -*-
# Author: Puyuan Du

class Radial:
    r'''Structure for data arranged by radials'''
    def __init__(self, data, drange, elev, reso, code, name,
                 scantime, dtype, stlon, stlat, lon=None, lat=None,
                 height=None, a_reso=None):
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
        '''
        self.data = data
        self.drange = drange
        self.elev = elev
        self.reso = reso
        self.code = code
        self.name = name
        self.scantime = scantime
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
            self.dtype.upper(), self.name, self.scantime, self.elev, self.drange)

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
    def __init__(self, data, xcor, ycor, scantime, code, name, dtype, **geoinfo):
        self.data = data
        self.xcor = xcor
        self.ycor = ycor
        self.geoinfo = geoinfo
        self.scantime = scantime
        self.code = code
        self.name = name
        self.dtype = dtype

class Grid:
    r'''Structure for processed grid data'''
    def __init__(self, data, drange, reso, code, name, scantime, dtype, lon, lat):
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

    def __repr__(self):
        repr_s = ('Datatype: {}\nStation name: {}\nScan time: {}\n')
        return repr_s.format(
            self.dtype.upper(), self.name, self.scantime)