# -*- coding: utf-8 -*-
# Author: Du puyuan

import os
from pathlib import Path
import warnings
import json
from typing import Union, Optional, Any, List

import numpy as np

from cinrad.visualize.basicfunc import (add_shp, save, setup_axes, setup_plot, text,
                                        change_cbar_text, draw_highlight_area, set_geoaxes)
from cinrad.constants import *
from cinrad.datastruct import Radial, Slice_, Grid
from cinrad.error import RadarPlotError
from cinrad.io.pup import _StormTrackInfo
from cinrad._typing import number_type

__all__ = ['PPI']

norm_plot = {'REF':norm1, 'VEL':norm2, 'CR':norm1, 'ET':norm5, 'VIL':norm1, 'RF':norm3,
             'ZDR':norm6, 'PHI':norm7, 'RHO':norm8, 'TREF':norm1, 'KDP':norm9, 'VILD':norm10} # Normalize object used to plot
norm_cbar = {'REF':norm1, 'VEL':norm4, 'CR':norm1, 'ET':norm4, 'VIL':norm4,
             'ZDR':norm4, 'PHI':norm4, 'RHO':norm4, 'TREF':norm1, 'KDP':norm4,
             'VILD':norm4} # Normalize object used for colorbar
cmap_plot = {'REF':r_cmap, 'VEL':v_cmap, 'CR':r_cmap, 'ET':et_cmap, 'VIL':vil_cmap, 'RF':rf_cmap,
             'ZDR':zdr_cmap, 'PHI':kdp_cmap, 'RHO':cc_cmap, 'TREF':r_cmap, 'KDP':kdp_cmap,
             'VILD':vil_cmap}
cmap_cbar = {'REF':r_cmap, 'VEL':v_cbar, 'CR':r_cmap, 'ET':et_cbar, 'VIL':vil_cbar,
             'ZDR':zdr_cbar, 'PHI':kdp_cbar, 'RHO':cc_cbar, 'TREF':r_cmap, 'KDP':kdp_cbar,
             'VILD':vil_cbar}
sec_plot = {'REF':r_cmap_smooth, 'VEL':v_cmap_smooth, 'ZDR':zdr_cmap_smooth, 'PHI':kdp_cmap_smooth, 'RHO':cc_cmap_smooth,
            'KDP':kdp_cmap_smooth}
prodname = {'REF':'Base Reflectivity', 'VEL':'Base Velocity', 'CR':'Composite Ref.',
            'ET':'Echo Tops', 'VIL':'V Integrated Liquid', 'ZDR':'Differential Ref.',
            'PHI':'Differential Phase', 'RHO':'Correlation Coe.', 'TREF':'Total Reflectivity',
            'KDP':'Spec. Diff. Phase', 'VILD':'VIL Density'}
unit = {'REF':'dBz', 'VEL':'m/s', 'CR':'dBz', 'ET':'km', 'VIL':'kg/m**2', 'ZDR':'dB', 'RHI':'deg',
        'RHO':'', 'TREF':'dBz', 'KDP':'deg/km', 'VILD':'g/m**3'}
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
                    '0.85', '0.8', '0.7', '0.6', '0.5', '0.3', '0.1', '0'],
             'TREF':None, 'KDP':['', '20', '7', '3.1', '2.4', '1.7', '1.1', '0.75', '0.5',
                                 '0.33', '0.22', '0.15', '0.1', '-0.1', '-0.2', '-0.4', '-0.8'],
             'VILD':['', '6', '5', '4', '3.5', '3', '2.5', '2.1', '1.8', '1.5', '1.2',
                     '0.9', '0.7', '0.5', '0.3', '0.1']}

def _prepare(data:Radial, datatype:str) -> tuple:
    if not data.geoflag:
        raise RadarPlotError('Geographic information should be contained in data')
    else:
        lon, lat = data.lon, data.lat
        r = data.data
    if data.dtype is not datatype:
        raise RadarPlotError('Expected datatype is "{}", received "{}"'.format(datatype, data.dtype))
    return lon, lat, r

class PPI(object):
    r'''
    Create a figure plotting plan position indicator

    Attributes
    ----------
    data: cinrad.datastruct.Radial / cinrad.datastruct.Grid
    settings: dict
        settings extracted from __init__ function
    geoax: cartopy.mpl.geoaxes.GeoAxes
        cartopy axes plotting georeferenced data
    fig: matplotlib.figure.Figure
    '''
    def __init__(self, data:Union[Radial, Grid], fig:Optional[Any]=None, norm:Optional[Any]=None,
                 cmap:Optional[Any]=None, nlabel:Optional[int]=None, label:Optional[List[str]]=None,
                 dpi:int=350, highlight:Optional[Union[str, List[str]]]=None, coastline:bool=False,
                 extent:Optional[List[number_type]]=None, add_slice:Optional[Slice_]=None,
                 style:str='black', add_city_names:bool=False, plot_labels:bool=True, **kwargs):
        self.data = data
        self.settings = {'cmap':cmap, 'norm':norm, 'nlabel':nlabel, 'label':label,
                         'highlight':highlight, 'coastline':coastline, 'path_customize':False,
                         'extent':extent, 'slice':add_slice, 'style':style, 'add_city_names':add_city_names,
                         'plot_labels':plot_labels}
        if fig is None:
            self.fig = setup_plot(dpi, style=style)
        else:
            self.fig = fig
        self._plot(**kwargs)

    def __call__(self, fpath:Optional[str]=None):
        if not fpath:
            fpath = os.path.join(str(Path.home()), 'PyCINRAD')
        else:
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

    def _plot(self, **kwargs):
        dtype = self.data.dtype
        lon, lat, var = _prepare(self.data, dtype)
        if self.settings['extent'] == None: #增加判断，城市名称绘制在选择区域内，否则自动绘制在data.lon和data.lat范围内
            self.settings['extent'] = [lon.min(), lon.max(), lat.min(), lat.max()]
        self.geoax = set_geoaxes(self.fig, extent=self.settings['extent'])
        if self.data.dtype in ['VEL', 'SW'] and self.data.include_rf:
            rf = var[1]
            var = var[0]
        popnan = var[np.logical_not(np.isnan(var))]
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        if self.data.dtype == 'CR':
            self.geoax.contourf(lon, lat, var, 128, norm=pnorm, cmap=pcmap, **kwargs)
        else:
            self.geoax.pcolormesh(lon, lat, var, norm=pnorm, cmap=pcmap, **kwargs)
            if self.data.dtype in ['VEL', 'SW'] and self.data.include_rf:
                self.geoax.pcolormesh(lon, lat, rf, norm=norm_plot['RF'], cmap=cmap_plot['RF'], **kwargs)
        add_shp(self.geoax, coastline=self.settings['coastline'], style=self.settings['style'],
                extent=self.settings['extent'])
        if self.settings['highlight']:
            draw_highlight_area(self.settings['highlight'])
        if self.settings['add_city_names']:
            self._add_city_names()
        ax2 = self.fig.add_axes([0.92, 0.06, 0.01, 0.35]) # axes used for text which has the same x-position as
                                                          # the colorbar axes (for matplotlib 3 compatibility)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax, cbar = setup_axes(self.fig, ccmap, cnorm)
        if not isinstance(clabel, type(None)):
            change_cbar_text(cbar, np.linspace(cnorm.vmin, cnorm.vmax, len(clabel)), clabel)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        if self.settings['plot_labels']:
            text(ax2, self.data.drange, self.data.reso, self.data.scantime, self.data.name, self.data.elev)
            ax2.text(0, 2.36, prodname[dtype], **plot_kw)
            ax2.text(0, 1.96, 'Max: {:.1f}{}'.format(np.max(popnan), unit[dtype]), **plot_kw)
            if self.data.dtype == 'VEL':
                ax2.text(0, 1.91, 'Min: {:.1f}{}'.format(np.min(popnan), unit[dtype]), **plot_kw)
        if self.settings['slice']:
            self.plot_cross_section(self.settings['slice'])

    def _save(self, fpath:str):
        if not self.settings['path_customize']:
            if not fpath.endswith(os.path.sep):
                fpath += os.path.sep
            if self.settings['slice']:
                data = self.settings['slice']
                stp = data.geoinfo['stp']
                enp = data.geoinfo['enp']
                sec = '_{}N{}E_{}N{}E'.format(stp[1], stp[0], enp[1], enp[0])
            else:
                sec = ''
            path_string = '{}{}_{}_{:.1f}_{}_{}{}.png'.format(fpath, self.data.code,
                                                              self.data.scantime.strftime('%Y%m%d%H%M%S'),
                                                              self.data.elev, self.data.drange,
                                                              self.data.dtype.upper(), sec)
        else:
            path_string = fpath
        save(path_string)

    def plot_range_rings(self, _range:Union[int, float, list], color:str='white', linewidth:number_type=0.5,
                         **kwargs):
        r'''Plot range rings on PPI plot.'''
        if isinstance(_range, (int, float)):
            _range = [_range]
        theta = np.linspace(0, 2 * np.pi, 800)
        for d in _range:
            radius = d / 111 # 1 degree = 111 km
            x, y = np.cos(theta) * radius + self.data.stp['lon'], np.sin(theta) * radius + self.data.stp['lat']
            #self.ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
            self.geoax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
            # add numbers for circle
            xText1, yText1 = -1 * radius + self.data.stp['lon'], self.data.stp['lat']
            xText2, yText2 = radius + self.data.stp['lon'], self.data.stp['lat']
            self.geoax.text(xText1, yText1,'{}'.format(d), fontsize=8)
            self.geoax.text(xText2, yText2,'{}'.format(d), fontsize=8)
        # add lines of 0 and 90 degree
        lenRadius = np.max(_range)
        x1, y1 = -1 * lenRadius + self.data.stp['lon'], self.data.stp['lat']
        x2, y2 = lenRadius + self.data.stp['lon'], self.data.stp['lat']
        self.geoax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, **kwargs)
        x3, y3 = np.cos(0.5 * np.pi) * lenRadius + self.data.stp['lon'], np.sin(0.5 * np.pi) * lenRadius + self.data.stp['lat']
        x4, y4 = np.cos(0.5 * np.pi) * lenRadius + self.data.stp['lon'], np.sin(1.5 * np.pi) * lenRadius + self.data.stp['lat']
        self.geoax.plot([x3, x4], [y3, y4], color=color, linewidth=linewidth, **kwargs)

    def plot_cross_section(self, data:Slice_, ymax:Optional[int]=None):
        r'''Plot cross section data below the PPI plot.'''
        if self.settings['style'] == 'black':
            linecolor = 'white'
        elif self.settings['stype'] == 'white':
            linecolor = 'black'
        self.settings['slice'] = data
        ax2 = self.fig.add_axes([0, -0.3, 0.9, 0.26])
        ax2.yaxis.set_ticks_position('right')
        ax2.set_xticks([])
        sl = data.data
        sl[sl == 0] = -1
        xcor = data.xcor
        ycor = data.ycor
        stp = data.geoinfo['stp']
        enp = data.geoinfo['enp']
        ax2.contourf(xcor, ycor, sl, 128, cmap=sec_plot[data.dtype], norm=norm_plot[data.dtype])
        if ymax:
            ax2.set_ylim(0, ymax)
        else:
            ax2.set_ylim(0, 15)
        ax2.set_title('Start: {}N {}E'.format(stp[1], stp[0]) + ' End: {}N {}E'.format(enp[1], enp[0]))
        self.geoax.plot([stp[0], enp[0]], [stp[1], enp[1]], marker='x', color=linecolor, zorder=5)

    def storm_track_info(self, filepath:str):
        r'''
        Add storm tracks from Nexrad Level III (PUP) STI product file
        '''
        sti = _StormTrackInfo(filepath)
        if len(sti.info.keys()) == 0:
            warnings.warn('No storm track to plot', RuntimeWarning)
            return
        else:
            stlist = sti.storm_list
            #extent = self.geoax.get_extent()
            for st in stlist:
                past = sti.track(st, 'past')
                fcs = sti.track(st, 'forecast')
                current = sti.current(st)
                if past:
                    self.geoax.plot(*past, marker='.', color='white', zorder=4, markersize=5)
                if fcs:
                    self.geoax.plot(*fcs, marker='+', color='white', zorder=4, markersize=5)
                self.geoax.scatter(*current, marker='o', s=15, zorder=5, color='lightgrey')
                #if (current[0] > extent[0]) and (current[0] < extent[1]) and (current[1] > extent[2]) and (current[1] < extent[3]):
                #    self.geoax.text(current[0] - 0.03, current[1] - 0.03, st, color='white', zorder=4)

    def gridlines(self, draw_labels:bool=True, linewidth:number_type=0, **kwargs):
        r'''Draw grid lines on cartopy axes'''
        liner = self.geoax.gridlines(draw_labels=draw_labels, linewidth=linewidth, **kwargs)
        liner.xlabels_top = False
        liner.ylabels_right = False

    def _add_city_names(self):
        with open(os.path.join(MODULE_DIR, 'data', 'chinaCity.json'), encoding='utf-8') as j:
            js = json.load(j)
        name = np.concatenate([[j['name'] for j in i['children']] for i in js])
        lon = np.concatenate([[j['log'] for j in i['children']] for i in js]).astype(float)
        lat = np.concatenate([[j['lat'] for j in i['children']] for i in js]).astype(float)
        extent = self.geoax.get_extent()
        fraction = (extent[1] - extent[0]) * 0.04
        target_city = (lon > (extent[0] + fraction)) & (lon < (extent[1] - fraction)) & (lat > (extent[2] + fraction)) & (lat < (extent[3] - fraction))
        for nm, stlon, stlat in zip(name[target_city], lon[target_city], lat[target_city]):
            self.geoax.text(stlon, stlat, nm, **plot_kw, color='darkgrey')