# -*- coding: utf-8 -*-
#Author: Du puyuan

from .basicfunc import add_shp, save, setup_axes, setup_plot, setup_basemap, text
from ..constants import font2, folderpath, norm4

import numpy as np

def base_reflectivity(data, smooth=False, draw_author=True):
    from ..constants import norm1, r_cmap
    if not data.geoflag:
        raise ValueError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        r = data.data
    if data.dtype is not 'r':
        raise ValueError('Expected datatype is "r", received "{}"'.format(data.dtype))
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

def base_velocity(data, draw_author=True):
    from ..constants import norm2, norm3, rf_cmap, v_cmap, v_cbar
    if not data.geoflag:
        raise ValueError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        if data.include_rf:
            v = data.data[0]
            rf = data.data[1]
        else:
            v = data.data
            rf = None
    if data.dtype is not 'v':
        raise ValueError('Expected datatype is "v", received "{}"'.format(data.dtype))
    fig = setup_plot(350)
    m = setup_basemap(lon, lat)
    m.pcolormesh(lon, lat, data, cmap=v_cmap, norm=norm2)
    if rf:
        m.pcolormesh(lon, lat, rf, cmap=rf_cmap, norm=norm3)
    add_shp(m)
    ax, cbar = setup_axes(fig, v_cbar, norm4)
    text(ax, data.drange, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Base Velocity', fontproperties=font2)
    ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(data.reso) , fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)

def echo_tops(data, draw_author=True):
    from ..constants import norm5, et_cmap, et_cbar
    if not data.geoflag:
        raise ValueError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        et = data.data
    if data.dtype is not 'et':
        raise ValueError('Expected datatype is "et", received "{}"'.format(data.dtype))
    fig = setup_plot(350)
    m = setup_basemap(lon, lat)
    dmax = et[np.logical_not(np.isnan(et))]
    m.pcolormesh(lon, lat, et, norm=norm5, cmap=et_cmap)
    add_shp(m)
    ax, cbar = setup_axes(fig, et_cbar, norm4)
    text(ax, data.drange, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Echo Tops', fontproperties=font2)
    ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(data.reso) , fontproperties=font2)
    ax.text(0, 1.81, 'Max: {:.1f}km'.format(np.max(dmax)), fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)