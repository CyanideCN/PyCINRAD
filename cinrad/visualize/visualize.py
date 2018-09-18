# -*- coding: utf-8 -*-
#Author: Du puyuan

from form_colormap import form_colormap
from .constants import deg2rad

import json
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cmx
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties

mpl.rc('font', family='Arial')
config = open(os.path.abspath('.') + '\\config.ini').read()
folderpath = json.loads(config)['filepath']
r_cmap = form_colormap('colormap\\r_main.txt', sep=True)
v_cmap = form_colormap('colormap\\v_main.txt', sep=False)
vel_cbar = form_colormap('colormap\\v_cbar.txt', sep=True)
rhi_cmap_smooth = form_colormap('colormap\\r_main.txt', sep=False, spacing='v')
r_cmap_smooth = form_colormap('colormap\\r_smooth.txt', sep=False, spacing='v')
zdr_cmap = form_colormap('colormap\\zdr_main.txt', sep=False)
zdr_cbar = form_colormap('colormap\\zdr_cbar.txt', sep=True)
kdp_cmap = form_colormap('colormap\\kdp_main.txt', sep=False)
kdp_cbar = form_colormap('colormap\\kdp_cbar.txt', sep=True)
cc_cmap = form_colormap('colormap\\cc_main.txt', sep=False)
cc_cbar = form_colormap('colormap\\cc_cbar.txt', sep=True)
et_cmap = form_colormap('colormap\\et_main.txt', sep=False)
et_cbar = form_colormap('colormap\\et_cbar.txt', sep=True)
vil_cmap = form_colormap('colormap\\vil_main.txt', sep=True)
vil_cbar = form_colormap('colormap\\vil_cbar.txt', sep=True)

font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)

class PPI():
    def __init__(self, filepath, radar_type=None):
        self.dpi = 350
        
    def _set_datatype(self, datatype):
        self.datatype = datatype

    def set_dpi(self, dpi):
        self.dpi = dpi

    def _set_cmap(self, cmap):
        self.cmap = cmap

    def _set_norm(self, norm):
        self.norm = norm

    def _setup_basemap(self):
        coor = self.projection(self.datatype)
        lons, lats = coor[0], coor[1]
        lonm, latm = np.max(lons), np.max(lats)
        x_delta = lonm - self.stationlon
        y_delta = latm - self.stationlat
        angle_offset = np.cos(self.elev * deg2rad)
        x_offset = x_delta / angle_offset
        y_offset = y_delta / angle_offset
        m = Basemap(llcrnrlon=self.stationlon - x_offset, urcrnrlon=self.stationlon + x_offset
                    , llcrnrlat=self.stationlat - y_offset, urcrnrlat=self.stationlat + y_offset, resolution="l")
        return lons, lats, m

    def _setup_plot(self):
        fig = plt.figure(figsize=(10, 10), dpi=self.dpi)
        plt.axis('off')
        plt.style.use('dark_background')
        return fig

    def _setup_axes(self, fig):
        ax = fig.add_axes([0.92, 0.12, 0.04, 0.35])
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=self.cmap, norm=self.norm, orientation='vertical', drawedges=False)
        cbar.ax.tick_params(labelsize=8)
        return ax, cbar

    def _text(self, ax, draw_author=True):
        ax.text(0, 2.09, 'Range: {:.0f}km'.format(self.drange), fontproperties=font2)
        ax.text(0, 2.01, 'Date: {}.{}.{}'.format(self.timestr[:4], self.timestr[4:6], self.timestr[6:8]), fontproperties=font2)
        ax.text(0, 1.97, 'Time: {}:{}'.format(self.timestr[8:10], self.timestr[10:12]), fontproperties=font2)
        ax.text(0, 1.93, 'RDA: ' + self.name, fontproperties=font2)
        ax.text(0, 1.89, 'Mode: Precipitation', fontproperties=font2)
        ax.text(0, 1.85, 'Elev: {:.2f}deg'.format(self.elev), fontproperties=font2)
        if draw_author:
            ax.text(0, 1.73, 'Made by HCl', fontproperties=font2)

    def _save(self):
        plt.savefig('{}{}_{}_{:.1f}_{}_{}.png'.format(
            folderpath, self.code, self.timestr, self.elev, self.drange, self.datatype.upper()), bbox_inches='tight', pad_inches = 0)
        plt.cla()

    def base_reflectivity(self, level, drange, draw_author=True, smooth=False, dpi=350):
        self._set_datatype('r')
        self._set_cmap(r_cmap)
        self._set_norm(norm1)
        data = self.reflectivity(level, drange)
        fig = self._setup_plot()
        lons, lats, m = self._setup_basemap()
        r1 = data[np.logical_not(np.isnan(data))]
        if smooth:
            self._set_cmap(r_cmap_smooth)
            m.contourf(lons.flatten(), lats.flatten(), data.flatten(), 256, cmap=r_cmap_smooth, norm=norm1, tri=True)
            suffix = '_smooth'
        else:
            data[data <= 2] = None
            m.pcolormesh(lons, lats, data, norm=norm1, cmap=r_cmap)
        _add_shp(m)
        ax, cbar = self._setup_axes(fig)
        self._text(ax, draw_author=draw_author)
        ax.text(0, 2.13, 'Base Reflectivity', fontproperties=font2)
        ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(self.Rreso) , fontproperties=font2)
        ax.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(r1)), fontproperties=font2)
        self._save()

    def base_velocity(self, level, drange, draw_author=True):
        self._set_datatype('v')
        self._set_cmap(vel_cbar)
        self._set_norm(cmx.Normalize(0, 1))
        data, rf = self.velocity(level, drange)
        fig = self._setup_plot()
        lons, lats, m = self._setup_basemap()
        m.pcolormesh(lons, lats, data, cmap=v_cmap, norm=norm2)
        rfmap = cmx.ListedColormap('#660066', '#FFFFFF')
        if rf is not None:
            m.pcolormesh(lons, lats, rf, cmap=rfmap, norm=cmx.Normalize(-1, 0))
        _add_shp(m)
        ax, cbar = self._setup_axes(fig)
        cbar.set_ticks(np.linspace(0, 1, 16))
        cbar.set_ticklabels(['RF', '', '27', '20', '15', '10', '5', '1', '0', '-1', '-5', '-10', '-15', '-20', '-27', '-35'])
        self._text(draw_author=draw_author)
        ax.text(0, 2.13, 'Base Velocity', fontproperties=font2)
        ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(self.Vreso) , fontproperties=font2)
        self._save()

    def echo_top(self, drange, draw_author=True):
        self._set_datatype('et')
        self._set_cmap(et_cbar)
        self._set_norm(cmx.Normalize(0, 1))
        data = self.echo_top(drange)
        self.set_elevation_angle(0)
        cbar.set_ticks(np.linspace(0, 1, 16))
        cbar.set_ticklabels(['', '21', '20', '18', '17', '15', '14', '12', '11', '9', '8', '6', '5', '3', '2', '0'])
        fig = self._setup_plot()
        lons, lats, m = self._setup_basemap()
        data[data > 25] = 0
        m.pcolormesh(lons, lats, data, cmap=et_cmap, norm=cmx.Normalize(0, 21))
        ax, cbar = self._setup_axes(fig)
        self._text(draw_author=draw_author)
        ax.text(0, 2.13, 'Echo Tops', fontproperties=font2)
        ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(1) , fontproperties=font2)
        ax.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(r1)), fontproperties=font2)
        _add_shp(m)
        self._save()

    def vert_integrated_liquid(self, drange, draw_author=True):
        self._set_datatype('vil')
        self._set_cmap(vil_cbar)
        self._set_norm(cmx.Normalize(0, 1))
        data = self.vert_integrated_liquid(drange=drange)
        self.set_elevation_angle(0)
        data[data <= 0] = np.nan

def _add_shp(m):
    m.readshapefile('shapefile\\County', 'states', drawbounds=True, linewidth=0.5, color='grey')
    m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.7, color='lightgrey')
    m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=1, color='white')