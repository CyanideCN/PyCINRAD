# -*- coding: utf-8 -*-
# Author: Du puyuan

import numpy as np
import pyart
import datetime
from tkinter import filedialog
from metpy.plots import colortables

from cinrad.datastruct import Radial
from cinrad.projection import get_coordinate
from cinrad.visualize import ppi
from cinrad.constants import deg2rad

name = filedialog.askopenfilename()
radar = pyart.io.read_nexrad_archive(name)
sweep = 1
drange = 300
irange = radar.get_start_end(sweep)
cali = pyart.correct.dealias_region_based(radar)
v = cali['data'][irange[0]:irange[1] + 1]
az = radar.get_azimuth(sweep) * deg2rad
az = np.append(az, az[0])
v_range = radar.range['data'] / 1000
reso = radar.range['meters_between_gates'] / 1000
v_ori = radar.get_field(sweep, 'velocity').T[:int(drange / reso)].T
slon, slat = radar.longitude['data'][0], radar.latitude['data'][0]
scantime = datetime.datetime.strptime(radar.time['units'], 'seconds since %Y-%m-%dT%H:%M:%SZ')
el = radar.get_elevation(sweep)[0]
lon, lat = get_coordinate(v_range[:int(drange / reso)], az, el, slon, slat)
timestr = scantime.strftime('%Y%m%d%H%M%S')
data = v.T[:int(drange / reso)].T
data_f = np.ma.array(data, mask=v_ori.mask)
data_f = np.concatenate((data_f, data_f[0, None]))
data_f[data_f == -64.5] = np.ma.masked
rf = np.ma.array(data_f, mask=(data_f != -64))
v_obj = Radial([data_f, rf], drange, el, reso, 'PGUA', 'PGUA', timestr, 'VEL', slon, slat)
v_obj.add_geoc(lon, lat, np.zeros(lon.shape))
norm2, v_cmap = colortables.get_with_range('NWS8bitVel', -64, 64)
fig = ppi.PPI(v_obj, coastline=True, norm=norm2, cmap=v_cmap, nlabel=17)
fig()
