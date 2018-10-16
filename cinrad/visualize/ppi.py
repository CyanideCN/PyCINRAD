# -*- coding: utf-8 -*-
# Author: Du puyuan

from .basicfunc import (add_shp, save, setup_axes, setup_plot, text
                        , change_cbar_text, draw_highlight_area, set_geoaxes)
from ..constants import *

import os
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

class PPI:
    r'''Create a figure plotting plan position indicator'''
    def __init__(self, data, norm=None, cmap=None, nlabel=None, label=None
                 , dpi=350, highlight=None, coastline=False):
        self.data = data
        self.settings = {'cmap':cmap, 'norm':norm, 'nlabel':nlabel, 'label':label, 'dpi':dpi
                         , 'highlight':highlight, 'coastline':coastline, 'path_customize':False}

    def __call__(self, *fpath):
        if not fpath:
            fpath = modpath
        else:
            fpath = fpath[0]
            if fpath.upper().endswith('.PNG'):
                self.settings['path_customize'] = True
            else:
                if not fpath.endswith(os.path.sep):
                    fpath += os.path.sep
        return self._plot(fpath)

    def _norm(self):
        if self.settings['norm']:
            n = self.settings['norm']
            if self.settings['label']:
                clabel = self.settings['label']
            else:
                nlabel = self.settings['nlabel']
                if nlabel:
                    clabel = np.linspace(n.vmin, n.vmax, nlabel).astype(str)
                else:
                    clabel = np.linspace(n.vmin, n.vmax, 10).astype(str)
            return n, n, clabel
        else:
            n = norm_plot[self.data.dtype]
            n2 = norm_cbar[self.data.dtype]
            return n, n2, cbar_text[self.data.dtype]

    def _cmap(self):
        if self.settings['cmap']:
            c = self.settings['cmap']
            return c, c
        else:
            c = cmap_plot[self.data.dtype]
            c2 = cmap_cbar[self.data.dtype]
            return c, c2

    def _plot(self, fpath):
        dtype = self.data.dtype
        lon, lat, var = _prepare(self.data, dtype)
        if self.data.dtype == 'v':
            rf = var[1]
            var = var[0]
        fig = setup_plot(self.settings['dpi'])
        geoax = set_geoaxes(lon, lat)
        popnan = var[np.logical_not(np.isnan(var))]
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        if self.data.dtype == 'cr':
            geoax.contourf(lon, lat, var, 128, norm=pnorm, cmap=pcmap)
        else:
            geoax.pcolormesh(lon, lat, var, norm=pnorm, cmap=pcmap)
            if self.data.dtype == 'v':
                geoax.pcolormesh(lon, lat, rf, norm=norm_plot['rf'], cmap=cmap_plot['rf'])
        add_shp(geoax, coastline=self.settings['coastline'])
        if self.settings['highlight']:
            draw_highlight_area(self.settings['highlight'])
        ax, cbar = setup_axes(fig, ccmap, cnorm)
        if not isinstance(clabel, type(None)):
            change_cbar_text(cbar, np.linspace(cnorm.vmin, cnorm.vmax, len(clabel)), clabel)
        text(ax, self.data.drange, self.data.reso, self.data.time, self.data.name, self.data.elev)
        ax.text(0, 2.13, prodname[dtype], fontproperties=font2)
        ax.text(0, 1.81, 'Max: {:.1f}{}'.format(np.max(popnan), unit[dtype]), fontproperties=font2)
        if self.data.dtype == 'v':
            ax.text(0, 1.77, 'Min: {:.1f}{}'.format(np.min(popnan), unit[dtype]), fontproperties=font2)
        if not self.settings['path_customize']:
            if not folderpath.endswith(os.path.sep):
                folderpath += os.path.sep
            path_string = '{}{}_{}_{:.1f}_{}_{}.png'.format(folderpath, code, timestr, elev, drange, datatype.upper())
        else:
            path_string = fpath
        save(path_string)