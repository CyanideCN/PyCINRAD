# -*- coding: utf-8 -*-
#Author: Du puyuan

from .basicfunc import (add_shp, save, setup_axes, setup_plot, text
                        , change_cbar_text, draw_highlight_area, set_geoaxes)
from ..constants import font2, norm4, folderpath

import numpy as np

def _prepare(data, datatype):
    if not data.geoflag:
        raise ValueError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        r = data.data
    if data.dtype is not datatype:
        raise ValueError('Expected datatype is "{}", received "{}"'.format(datatype, data.dtype))
    return lon, lat, data.data

def base_reflectivity(data, smooth=False, draw_author=True, highlight=None):
    from ..constants import norm1, r_cmap
    lon, lat, r = _prepare(data, 'r')
    fig = setup_plot(350)
    renderer = set_geoaxes(lon, lat)
    dmax = r[np.logical_not(np.isnan(r))]
    if smooth:
        renderer.tricontourf(lons.flatten(), lats.flatten(), r.flatten(), 256, cmap=r_cmap_smooth
                    , norm=norm1)
    else:
        r[r <= 2] = None
        renderer.pcolormesh(lon, lat, r, norm=norm1, cmap=r_cmap)
    add_shp(renderer)
    if highlight:
        draw_highlight_area(highlight)
    ax, cbar = setup_axes(fig, r_cmap, norm1)
    text(ax, data.drange, data.reso, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Base Reflectivity', fontproperties=font2)
    ax.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(dmax)), fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)

def base_velocity(data, draw_author=True, highlight=None):
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
    if data.dtype is not 'v':
        raise ValueError('Expected datatype is "v", received "{}"'.format(data.dtype))
    fig = setup_plot(350)
    renderer = set_geoaxes(lon, lat)
    renderer.pcolormesh(lon, lat, v, cmap=v_cmap, norm=norm2)
    if data.include_rf:
        renderer.pcolormesh(lon, lat, rf, cmap=rf_cmap, norm=norm3)
    add_shp(renderer)
    if highlight:
        draw_highlight_area(highlight)
    ax, cbar = setup_axes(fig, v_cbar, norm4)
    change_cbar_text(cbar, np.linspace(0, 1, 16), ['RF', '', '27', '20', '15', '10', '5', '1', '0'
                                                   , '-1', '-5', '-10', '-15', '-20', '-27', '-35'])
    text(ax, data.drange, data.reso, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Base Velocity', fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)

def echo_tops(data, draw_author=True, highlight=None):
    from ..constants import norm5, et_cmap, et_cbar
    lon, lat, et = _prepare(data, 'et')
    fig = setup_plot(350)
    renderer = set_geoaxes(lon, lat)
    dmax = et[np.logical_not(np.isnan(et))]
    renderer.pcolormesh(lon, lat, et, norm=norm5, cmap=et_cmap)
    add_shp(renderer)
    if highlight:
        draw_highlight_area(highlight)
    ax, cbar = setup_axes(fig, et_cbar, norm4)
    change_cbar_text(cbar, np.linspace(0, 1, 16), ['', '21', '20', '18', '17', '15', '14', '12'
                                                   , '11', '9', '8', '6', '5', '3', '2', '0'])
    text(ax, data.drange, data.reso, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Echo Tops', fontproperties=font2)
    ax.text(0, 1.81, 'Max: {:.1f}km'.format(np.max(dmax)), fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)

def vert_integrated_liquid(data, draw_author=True, highlight=None):
    from ..constants import norm1, vil_cmap, vil_cbar
    lon, lat,vil = _prepare(data, 'vil')
    fig = setup_plot(350)
    renderer = set_geoaxes(lon, lat)
    dmax = vil[np.logical_not(np.isnan(vil))]
    vil[vil <= 0] = None
    renderer.pcolormesh(lon, lat, vil, norm=norm1, cmap=vil_cmap)
    add_shp(renderer)
    if highlight:
        draw_highlight_area(highlight)
    ax, cbar = setup_axes(fig, vil_cbar, norm4)
    change_cbar_text(cbar, np.linspace(0, 1, 16), ['', '70', '65', '60', '55', '50', '45', '40'
                                                   , '35', '30' , '25', '20', '15', '10', '5', '0'])
    text(ax, data.drange, data.reso, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'V Integrated Liquid', fontproperties=font2)
    ax.text(0, 1.81, 'Max: {:.1f}kg/m**2'.format(np.max(dmax)), fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)

def composite_reflectivity(data, draw_author=True, highlight=None):
    from ..constants import norm1, r_cmap
    lon, lat, r = _prepare(data, 'cr')
    fig = setup_plot(350)
    renderer = set_geoaxes(lon, lat)
    dmax = r[np.logical_not(np.isnan(r))]
    r[r <= 2] = None
    renderer.contourf(lon, lat, r, 128, norm=norm1, cmap=r_cmap)
    add_shp(renderer)
    if highlight:
        draw_highlight_area(highlight)
    ax, cbar = setup_axes(fig, r_cmap, norm1)
    text(ax, data.drange, data.reso, data.time, data.name, data.elev, draw_author=draw_author)
    ax.text(0, 2.13, 'Composite Ref.', fontproperties=font2)
    ax.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(dmax)), fontproperties=font2)
    save(folderpath, data.code, data.time, data.elev, data.drange, data.dtype)