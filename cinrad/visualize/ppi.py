# -*- coding: utf-8 -*-
# Author: Du puyuan

from .basicfunc import (add_shp, save, setup_axes, setup_plot, text
                        , change_cbar_text, draw_highlight_area, set_geoaxes)
from ..constants import *
from ..error import RadarPlotError

import os
from pathlib import Path
import numpy as np

norm_plot = {'REF':norm1, 'VEL':norm2, 'CR':norm1, 'ET':norm5, 'VIL':norm1, 'RF':norm3,
             'ZDR':norm6, 'PHI':norm7, 'RHO':norm8} # Normalize object used to plot
norm_cbar = {'REF':norm1, 'VEL':norm4, 'CR':norm1, 'ET':norm4, 'VIL':norm4,
             'ZDR':norm4, 'PHI':norm4, 'RHO':norm4} # Normalize object used for colorbar
cmap_plot = {'REF':r_cmap, 'VEL':v_cmap, 'CR':r_cmap, 'ET':et_cmap, 'VIL':vil_cmap, 'RF':rf_cmap,
             'ZDR':zdr_cmap, 'PHI':kdp_cmap, 'RHO':cc_cmap}
cmap_cbar = {'REF':r_cmap, 'VEL':v_cbar, 'CR':r_cmap, 'ET':et_cbar, 'VIL':vil_cbar,
             'ZDR':zdr_cbar, 'PHI':kdp_cmap, 'RHO':cc_cbar}
prodname = {'REF':'Base Reflectivity', 'VEL':'Base Velocity', 'CR':'Composite Ref.',
            'ET':'Echo Tops', 'VIL':'V Integrated Liquid', 'ZDR':'Differential Ref.',
            'PHI':'Difference Phase', 'RHO':'Correlation Coe.'}
unit = {'REF':'dBz', 'VEL':'m/s', 'CR':'dBz', 'ET':'km', 'VIL':'kg/m**2', 'ZDR':'dB', 'RHI':'deg',
        'RHO':''}
cbar_text = {'REF':None, 'VEL':['RF', '', '27', '20', '15', '10', '5', '1', '0',
                                '-1', '-5', '-10', '-15', '-20', '-27', '-35'],
             'CR':None, 'ET':['', '21', '20', '18', '17', '15', '14', '12',
                              '11', '9', '8', '6', '5', '3', '2', '0'],
             'VIL':['', '70', '65', '60', '55', '50', '45', '40', '35', '30',
                    '25', '20', '15', '10', '5', '0'],
             'ZDR':['', '5', '4', '3.5', '3', '2.5', '2', '1.5', '1', '0.8', '0.5',
                    '0.2', '0', '-1', '-2', '-3', '-4'],
             'PHI':np.linspace(360, 260, 17).astype(str),
             'RHO':['', '0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '0.92', '0.9',
                    '0.85', '0.8', '0.7', '0.6', '0.5', '0.3', '0.1', '0']}

def _prepare(data, datatype):
    if not data.geoflag:
        raise RadarPlotError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        r = data.data
    if data.dtype is not datatype:
        raise RadarPlotError('Expected datatype is "{}", received "{}"'.format(datatype, data.dtype))
    return lon, lat, data.data

class PPI:
    r'''Create a figure plotting plan position indicator'''
    def __init__(self, data, norm=None, cmap=None, nlabel=None, label=None
                 , dpi=350, highlight=None, coastline=False, extent=None):
        self.data = data
        self.settings = {'cmap':cmap, 'norm':norm, 'nlabel':nlabel, 'label':label, 'dpi':dpi,
                         'highlight':highlight, 'coastline':coastline, 'path_customize':False,
                         'extent':extent}
        self.ax = self._plot()

    def __call__(self, *fpath):
        if not fpath:
            fpath = os.path.join(str(Path.home()), 'PyCINRAD')
        else:
            fpath = fpath[0]
            if fpath.upper().endswith('.PNG'):
                self.settings['path_customize'] = True
            else:
                if not fpath.endswith(os.path.sep):
                    fpath += os.path.sep
        return self._save(fpath)

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

    def _plot(self):
        dtype = self.data.dtype
        lon, lat, var = _prepare(self.data, dtype)
        if self.data.dtype == 'VEL' and self.data.include_rf:
            rf = var[1]
            var = var[0]
        fig = setup_plot(self.settings['dpi'])
        geoax = set_geoaxes(lon, lat, extent=self.settings['extent'])
        popnan = var[np.logical_not(np.isnan(var))]
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        if self.data.dtype == 'CR':
            geoax.contourf(lon, lat, var, 128, norm=pnorm, cmap=pcmap)
        else:
            geoax.pcolormesh(lon, lat, var, norm=pnorm, cmap=pcmap)
            if self.data.dtype == 'VEL' and self.data.include_rf:
                geoax.pcolormesh(lon, lat, rf, norm=norm_plot['RF'], cmap=cmap_plot['RF'])
        add_shp(geoax, coastline=self.settings['coastline'])
        if self.settings['highlight']:
            draw_highlight_area(self.settings['highlight'])
        ax, cbar = setup_axes(fig, ccmap, cnorm)
        if not isinstance(clabel, type(None)):
            change_cbar_text(cbar, np.linspace(cnorm.vmin, cnorm.vmax, len(clabel)), clabel)
        text(ax, self.data.drange, self.data.reso, self.data.time, self.data.name, self.data.elev)
        ax.text(0, 2.13, prodname[dtype], fontproperties=font2)
        ax.text(0, 1.81, 'Max: {:.1f}{}'.format(np.max(popnan), unit[dtype]), fontproperties=font2)
        if self.data.dtype == 'VEL':
            ax.text(0, 1.77, 'Min: {:.1f}{}'.format(np.min(popnan), unit[dtype]), fontproperties=font2)
        return geoax

    def _save(self, fpath):
        if not self.settings['path_customize']:
            if not fpath.endswith(os.path.sep):
                fpath += os.path.sep
            path_string = '{}{}_{}_{:.1f}_{}_{}.png'.format(fpath, self.data.code, self.data.time,
                                                            self.data.elev, self.data.drange,
                                                            self.data.dtype.upper())
        else:
            path_string = fpath
        save(path_string)

    def plot_range_rings(self, _range, color='white', linewidth=0.5, **kwargs):
        if isinstance(_range, int):
            _range = [_range]
        theta = np.linspace(0, 2 * np.pi, 800)
        for d in _range:
            radius = d / 111
            x, y = np.cos(theta) * radius + self.data.stp['lon'], np.sin(theta) * radius + self.data.stp['lat']
            self.ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)