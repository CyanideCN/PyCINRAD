# -*- coding: utf-8 -*-
#Author: Du puyuan

from ..constants import font2, modpath
from ..error import RadarPlotError
from .shapepatch import highlight_area

import os

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from cartopy.io import shapereader
import cartopy.crs as ccrs

def setup_plot(dpi, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    plt.style.use('dark_background')
    return fig

def setup_axes(fig, cmap, norm):
    ax = fig.add_axes([0.92, 0.12, 0.04, 0.35])
    cbar = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', drawedges=False)
    cbar.ax.tick_params(labelsize=8)
    return ax, cbar

def text(ax, drange, reso, timestr, name, elev, draw_author=True):
    ax.text(0, 2.09, 'Range: {:.0f}km'.format(drange), fontproperties=font2)
    ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(reso) , fontproperties=font2)
    ax.text(0, 2.01, 'Date: {}.{}.{}'.format(timestr[:4], timestr[4:6], timestr[6:8]), fontproperties=font2)
    ax.text(0, 1.97, 'Time: {}:{}'.format(timestr[8:10], timestr[10:12]), fontproperties=font2)
    ax.text(0, 1.93, 'RDA: ' + name, fontproperties=font2)
    ax.text(0, 1.89, 'Mode: Precipitation', fontproperties=font2)
    ax.text(0, 1.85, 'Elev: {:.2f}deg'.format(elev), fontproperties=font2)
    if draw_author:
        ax.text(0, 1.73, 'Made by HCl', fontproperties=font2)

def save(folderpath, code, timestr, elev, drange, datatype):
    if not folderpath.endswith(os.path.sep):
        folderpath += os.path.sep
    plt.savefig('{}{}_{}_{:.1f}_{}_{}.png'.format(
        folderpath, code, timestr, elev, drange, datatype.upper()), bbox_inches='tight', pad_inches = 0)
    plt.cla()

def add_shp(renderer):
    root = os.path.join(modpath, 'shapefile')
    flist = [os.path.join(root, i) for i in ['County', 'City', 'Province']]
    shps = [shapereader.Reader(i).geometries() for i in flist]
    renderer.add_geometries(shps[0], ccrs.PlateCarree(), edgecolor='grey', facecolor='None', zorder=1, linewidth=0.5)
    renderer.add_geometries(shps[1], ccrs.PlateCarree(), edgecolor='lightgrey', facecolor='None', zorder=1, linewidth=0.7)
    renderer.add_geometries(shps[2], ccrs.PlateCarree(), edgecolor='white', facecolor='None', zorder=1, linewidth=1)

def change_cbar_text(cbar, tick, text):
    cbar.set_ticks(tick)
    cbar.set_ticklabels(text)

def draw_highlight_area(area):
    patch = highlight_area(area)
    ax_ = plt.gca()
    pat = ax_.add_patch(patch)
    pat.set_zorder(2)

def set_geoaxes(lon, lat):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.background_patch.set_fill(False)
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], ccrs.PlateCarree())
    return ax