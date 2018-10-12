# -*- coding: utf-8 -*-
# Author: Du puyuan

from .basicfunc import (add_shp, save, setup_axes, setup_plot, text
                        , change_cbar_text, draw_highlight_area, set_geoaxes)
from ..constants import *

import numpy as np

norm_plot = {'r':norm1, 'v':norm2, 'cr':norm1, 'et':norm5, 'vil':norm1, 'rf':norm3} # Normalize object used to plot
norm_cbar = {'r':norm1, 'v':norm4, 'cr':norm1, 'et':norm4, 'vil':norm4} # Normalize object used for colorbar
cmap_plot = {'r':r_cmap, 'v':v_cmap, 'cr':r_cmap, 'et':et_cmap, 'vil':vil_cmap, 'rf':rf_cmap}
cmap_cbar = {'r':r_cmap, 'v':v_cbar, 'cr':r_cmap, 'et':et_cbar, 'vil':vil_cbar}
prodname = {'r':'Base Reflectivity', 'v':'Base Velocity', 'cr':'Composite Ref.',
            'et':'Echo Tops', 'vil':'V Integrated Liquid'}
unit = {'r':'dBz', 'v':'m/s', 'cr':'dBz', 'et':'km', 'vil':'kg/m**2'}
cbar_text = {'r':None, 'v':['RF', '', '27', '20', '15', '10', '5', '1', '0'
                            , '-1', '-5', '-10', '-15', '-20', '-27', '-35'],
             'cr':None, 'et':['', '21', '20', '18', '17', '15', '14', '12'
                              , '11', '9', '8', '6', '5', '3', '2', '0'],
             'vil':['', '70', '65', '60', '55', '50', '45', '40', '35', '30'
                    , '25', '20', '15', '10', '5', '0']}

def _prepare(data, datatype):
    if not data.geoflag:
        raise ValueError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        r = data.data
    if data.dtype is not datatype:
        raise ValueError('Expected datatype is "{}", received "{}"'.format(datatype, data.dtype))
    return lon, lat, data.data

class Display:
    def __init__(self, data, norm=None, cmap=None, dpi=350, draw_author=True, highlight=None, coastline=False):
        self.data = data
        if not norm:
            self.norm = norm_plot[data.dtype]
        else:
            self.norm = norm
        if not cmap:
            self.cmap = cmap_plot[data.dtype]
        else:
            self.cmap = cmap
        self.settings = {'dpi':dpi, 'draw_author':draw_author, 'highlight':highlight, 'coastline':coastline}

    def __call__(self, **fpath):
        if not fpath.keys():
            fpath = modpath
        return self._plot(fpath)

    def _plot(self, fpath):
        dtype = self.data.dtype
        lon, lat, var = _prepare(self.data, dtype)
        if self.data.dtype == 'v':
            rf = var[1]
            var = var[0]
        fig = setup_plot(self.settings['dpi'])
        geoax = set_geoaxes(lon, lat)
        popnan = var[np.logical_not(np.isnan(var))]
        if self.data.dtype == 'cr':
            geoax.contourf(lon, lat, var, 128, norm=norm_plot[dtype], cmap=cmap_plot[dtype])
        else:
            geoax.pcolormesh(lon, lat, var, norm=norm_plot[dtype], cmap=cmap_plot[dtype])
            if self.data.dtype == 'v':
                geoax.pcolormesh(lon, lat, rf, norm=norm_plot['rf'], cmap=cmap_plot['rf'])
        add_shp(geoax, coastline=self.settings['coastline'])
        if self.settings['highlight']:
            draw_highlight_area(self.settings['highlight'])
        ax, cbar = setup_axes(fig, cmap_cbar[dtype], norm_cbar[dtype])
        text_ = cbar_text[dtype]
        if text_:
            change_cbar_text(cbar, np.linspace(0, 1, len(text_)), text_)
        text(ax, self.data.drange, self.data.reso, self.data.time, self.data.name, self.data.elev
             , draw_author=self.settings['draw_author'])
        ax.text(0, 2.13, prodname[dtype], fontproperties=font2)
        ax.text(0, 1.81, 'Max: {:.1f}{}'.format(np.max(popnan), unit[dtype]), fontproperties=font2)
        if self.data.dtype == 'v':
            ax.text(0, 1.77, 'Min: {:.1f}{}'.format(np.min(popnan), unit[dtype]), fontproperties=font2)
        save(fpath, self.data.code, self.data.time, self.data.elev, self.data.drange, self.data.dtype)