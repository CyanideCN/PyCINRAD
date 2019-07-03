# -*- coding: utf-8 -*-
# Author: Du puyuan

import os
from datetime import datetime
from typing import Union, Optional, Any, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
from cartopy.io import shapereader
import cartopy.crs as ccrs
import shapefile

from cinrad.constants import MODULE_DIR
from cinrad._typing import Array_T, Number_T
from cinrad.error import RadarPlotError

class ShpReader(shapereader.BasicReader):
    r'''Customized reader to deal with encoding issue'''
    def __init__(self, filename:str, encoding:str='gbk'):
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

def setup_plot(dpi:Number_T, figsize:tuple=(9, 9), style:str='black') -> Any:
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

def text(ax:Any, drange:Number_T, reso:float, scantime:datetime, name:str, task:str, elev:float):
    from cinrad.constants import plot_kw
    ax.text(0, 2.31, 'Range: {:.0f}km'.format(drange), **plot_kw)
    ax.text(0, 2.26, 'Resolution: {:.2f}km'.format(reso) , **plot_kw)
    ax.text(0, 2.21, 'Date: {}'.format(scantime.strftime('%Y.%m.%d')), **plot_kw)
    ax.text(0, 2.16, 'Time: {}'.format(scantime.strftime('%H:%M')), **plot_kw)
    if name is None:
        name = 'Unknown'
    ax.text(0, 2.11, 'RDA: ' + name, **plot_kw)
    ax.text(0, 2.06, 'Task: {}'.format(task), **plot_kw)
    ax.text(0, 2.01, 'Elev: {:.2f}deg'.format(elev), **plot_kw)

def save(fpath:str):
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def add_shp(ax:Any, proj:ccrs.Projection, coastline:bool=False, style:str='black',
            extent:Optional[Array_T]=None):
    shp_crs = ccrs.PlateCarree()
    root = os.path.join(MODULE_DIR, 'data', 'shapefile')
    flist = [os.path.join(root, i) for i in ['County', 'City', 'Province']]
    shps = [ShpReader(i).geometries() for i in flist]
    if style == 'black':
        line_colors = ['grey', 'lightgrey', 'white']
    elif style == 'white':
        line_colors = ['lightgrey', 'grey', 'black']
    ax.add_geometries(shps[0], shp_crs, edgecolor=line_colors[0], facecolor='None', zorder=3, linewidth=0.5)
    ax.add_geometries(shps[1], shp_crs, edgecolor=line_colors[1], facecolor='None', zorder=3, linewidth=0.7)
    ax.add_geometries(shps[2], shp_crs, edgecolor=line_colors[2], facecolor='None', zorder=3, linewidth=1)
    if coastline:
        ax.coastlines(resolution='10m', color=line_colors[2], zorder=3, linewidth=1)

def change_cbar_text(cbar:ColorbarBase, tick:list, text:list):
    cbar.set_ticks(tick)
    cbar.set_ticklabels(text)

def highlight_area(area:Union[Array_T, str], linecolor:str='red', **kwargs) -> List[Line2D]:
    r'''Return list of Line2D object for given area name'''
    fpath = os.path.join(MODULE_DIR, 'data', 'shapefile', 'City')
    shp = shapefile.Reader(fpath, encoding='gbk')
    rec = shp.shapeRecords()
    lines = list()
    if isinstance(area, str):
        area = [area]
    for i in area:
        if not isinstance(i, str):
            raise RadarPlotError('Area name should be str')
        name = np.array([i.record[2] for i in rec])
        target = np.array(rec)[(name == i).nonzero()[0]]
        for j in target:
            pts = j.shape.points
            x = [i[0] for i in pts]
            y = [i[1] for i in pts]
            lines.append(Line2D(x, y, color=linecolor))
    return lines

def draw_highlight_area(area:Union[Array_T, str]):
    lines = highlight_area(area)
    ax_ = plt.gca()
    for l in lines:
        pat = ax_.add_artist(l)
        pat.set_zorder(4)

def set_geoaxes(fig:Any, proj:ccrs.Projection, extent:Array_T) -> Any:
    ax = fig.add_axes([0, 0, 0.9, 0.9], projection=proj)
    ax.background_patch.set_fill(False)
    x_min, x_max, y_min, y_max = extent[0], extent[1], extent[2], extent[3]
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())
    return ax