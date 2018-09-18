# -*- coding: utf-8 -*-
#Author: Du puyuan

from .basicfunc import add_shp, save, setup_axes, setup_plot, setup_basemap, text
from ..constants import font2, folderpath

import numpy as np

def base_reflectivity(data, smooth=False, draw_author=True):
    from ..constants import norm1, r_cmap
    if not data.geoflag:
        raise ValueError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        r = data.data
    fig = setup_plot(350)
    m = setup_basemap(lon, lat)
    dmax = r[np.logical_not(np.isnan(r))]
    if smooth:
        m.contourf(lons.flatten(), lats.flatten(), r.flatten(), 256, cmap=r_cmap_smooth, norm=norm1, tri=True)
    else:
        r[r <= 2] = None
        m.pcolormesh(lon, lat, r, norm=norm1, cmap=r_cmap)
    add_shp(m)
    ax, cbar = setup_axes(fig, r_cmap, norm1)
    text(ax, data.drange, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Base Reflectivity', fontproperties=font2)
    ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(data.reso) , fontproperties=font2)
    ax.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(dmax)), fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)