# -*- coding: utf-8 -*-
# Author: Du puyuan

import os
from datetime import datetime
from typing import Union, Optional, Any

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from cartopy.io import shapereader
import cartopy.crs as ccrs
import shapefile

from cinrad.constants import MODULE_DIR, plot_kw
from cinrad.visualize.shapepatch import highlight_area
from cinrad._typing import GList, number_type

class ShpReader(shapereader.BasicReader):
    r'''Customized reader to deal with encoding issue'''
    def __init__(self, filename, encoding='gbk'):
        # Validate the filename/shapefile
        self._reader = reader = shapefile.Reader(filename, encoding=encoding)
        if reader.shp is None or reader.shx is None or reader.dbf is None:
            raise ValueError("Incomplete shapefile definition "
                             "in '%s'." % filename)

        # Figure out how to make appropriate shapely geometry instances
        shapeType = reader.shapeType
        self._geometry_factory = shapereader.GEOMETRY_FACTORIES.get(shapeType)
        if self._geometry_factory is None:
            raise ValueError('Unsupported shape type: %s' % shapeType)

        self._fields = self._reader.fields

def setup_plot(dpi:number_type, figsize:tuple=(9, 9), style:str='black') -> Any:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    if style == 'black':
        plt.style.use('dark_background')
    return fig

def setup_axes(fig:Any, cmap:Any, norm:Any) -> tuple:
    ax = fig.add_axes([0.92, 0.06, 0.045, 0.38])
    cbar = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', drawedges=False)
    cbar.ax.tick_params(axis='both', which='both', length=0, labelsize=10)
    cbar.outline.set_visible(False)
    return ax, cbar

def text(ax:Any, drange:number_type, reso:float, scantime:datetime, name:str, elev:float):
    ax.text(0, 2.31, 'Range: {:.0f}km'.format(drange), **plot_kw)
    ax.text(0, 2.26, 'Resolution: {:.2f}km'.format(reso) , **plot_kw)
    ax.text(0, 2.21, 'Date: {}'.format(scantime.strftime('%Y.%m.%d')), **plot_kw)
    ax.text(0, 2.16, 'Time: {}'.format(scantime.strftime('%H:%M')), **plot_kw)
    if name is None:
        name = 'Unknown'
    ax.text(0, 2.11, 'RDA: ' + name, **plot_kw)
    ax.text(0, 2.06, 'Mode: Precipitation', **plot_kw)
    ax.text(0, 2.01, 'Elev: {:.2f}deg'.format(elev), **plot_kw)

def save(fpath:str):
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def add_shp(ax:Any, coastline:bool=False, style:str='black', extent:Optional[GList]=None):
    root = os.path.join(MODULE_DIR, 'data', 'shapefile')
    flist = [os.path.join(root, i) for i in ['County', 'City', 'Province']]
    shps = [ShpReader(i).geometries() for i in flist]
    if style == 'black':
        line_colors = ['grey', 'lightgrey', 'white']
    elif style == 'white':
        line_colors = ['lightgrey', 'grey', 'black']
    ax.add_geometries(shps[0], ccrs.PlateCarree(), edgecolor=line_colors[0], facecolor='None', zorder=3, linewidth=0.5)
    ax.add_geometries(shps[1], ccrs.PlateCarree(), edgecolor=line_colors[1], facecolor='None', zorder=3, linewidth=0.7)
    ax.add_geometries(shps[2], ccrs.PlateCarree(), edgecolor=line_colors[2], facecolor='None', zorder=3, linewidth=1)
    if coastline:
        ax.coastlines(resolution='10m', color=line_colors[2], zorder=3, linewidth=1)

def change_cbar_text(cbar:ColorbarBase, tick:list, text:list):
    cbar.set_ticks(tick)
    cbar.set_ticklabels(text)

def draw_highlight_area(area:Union[GList, str]):
    lines = highlight_area(area)
    ax_ = plt.gca()
    for l in lines:
        pat = ax_.add_artist(l)
        pat.set_zorder(4)

def set_geoaxes(fig:Any, extent:GList) -> Any:
    ax = fig.add_axes([0, 0, 0.9, 0.9], projection=ccrs.PlateCarree())
    ax.background_patch.set_fill(False)
    x_min, x_max, y_min, y_max = extent[0], extent[1], extent[2], extent[3]
    ax.set_extent([x_min, x_max, y_min, y_max], ccrs.PlateCarree())
    return ax