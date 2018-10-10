# -*- coding: utf-8 -*-
# Author: Du puyuan

import numpy as np
from metpy.io.nexrad import Level2File
from tkinter import filedialog

from cinrad.datastruct import V
from cinrad.projection import get_coordinate
from cinrad.visualize import ppi
from cinrad.constants import deg2rad

name = filedialog.askopenfilename()
f = Level2File(name)
# Pull data out of the file
sweep = 1
range = 230
# First item in ray is header, which has azimuth angle
az = np.array([ray[0].az_angle for ray in f.sweeps[sweep]])
el = f.sweeps[sweep][0][0].el_angle
# 5th item is a dict mapping a var name (byte string) to a tuple
# of (header, data array)
v_hdr = f.sweeps[sweep][0][4][b'VEL'][0]
reso = v_hdr.gate_width
v_range = np.arange(v_hdr.num_gates) * reso + v_hdr.first_gate
v = np.array([ray[4][b'VEL'][1] for ray in f.sweeps[sweep]])
slon = f.sweeps[0][0][1].lon
slat = f.sweeps[0][0][1].lat
lon, lat = get_coordinate(v_range[:int(range / reso)], az * deg2rad, el, slon, slat)
timestr = f.dt.strftime('%Y%m%d%H%M%S')
data = v.T[:int(range / reso)].T
v_obj = V(data, range, el, reso, f.stid.decode(), f.stid.decode()
          , timestr, slon, slat, include_rf=False)
v_obj.add_geoc(lon, lat, np.zeros(lon.shape))
ppi.base_velocity(v_obj, coastline=True, lscale=True)