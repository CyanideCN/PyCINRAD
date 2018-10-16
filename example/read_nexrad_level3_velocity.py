# -*- coding: utf-8 -*-
# Author: Du puyuan

import numpy as np
from metpy.io.nexrad import Level3File
from tkinter import filedialog
from metpy.plots import colortables

from cinrad.datastruct import V
from cinrad.projection import get_coordinate
from cinrad.visualize import ppi

name = filedialog.askopenfilename()
f = Level3File(name)
# Pull the data out of the file object
datadict = f.sym_block[0][0]
# Turn into an array, then mask
data = np.array(datadict['data']).astype(float)
data[data == 0] = np.nan
rf = np.ma.array(data, mask=(data != 1))
# Grab azimuths and calculate a range based on number of gates
az = np.deg2rad(datadict['start_az'] + [datadict['end_az'][-1]])
rng = np.linspace(0, f.max_range, data.shape[-1] + 1)
elev = f.metadata['el_angle']
slon, slat = f.lon, f.lat
scantime = f.metadata['msg_time']
lon, lat = get_coordinate(rng, az, elev, slon, slat)
threshold = f.thresholds
data = data * threshold[1] / 10 + threshold[0] / 10
v_obj = V([data, rf], int(f.max_range), elev, f.ij_to_km, f.siteID, f.siteID
          , scantime.strftime('%Y%m%d%H%M%S'), slon, slat)
v_obj.add_geoc(lon, lat, np.zeros(lon.shape))
norm2, v_cmap = colortables.get_with_range('NWS8bitVel', -64, 64)
fig = ppi.PPI(v_obj, coastline=True, norm=norm2, cmap=v_cmap, nlabel=17)
fig()