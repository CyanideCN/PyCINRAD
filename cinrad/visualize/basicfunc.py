# -*- coding: utf-8 -*-
#Author: Du puyuan

from ..constants import font2, modpath

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.basemap import Basemap

def setup_plot(dpi):
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    plt.axis('off')
    plt.style.use('dark_background')
    return fig

def setup_axes(fig, cmap, norm):
    ax = fig.add_axes([0.92, 0.12, 0.04, 0.35])
    cbar = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', drawedges=False)
    cbar.ax.tick_params(labelsize=8)
    return ax, cbar

def text(ax, drange, timestr, name, elev, draw_author=True):
    ax.text(0, 2.09, 'Range: {:.0f}km'.format(drange), fontproperties=font2)
    ax.text(0, 2.01, 'Date: {}.{}.{}'.format(timestr[:4], timestr[4:6], timestr[6:8]), fontproperties=font2)
    ax.text(0, 1.97, 'Time: {}:{}'.format(timestr[8:10], timestr[10:12]), fontproperties=font2)
    ax.text(0, 1.93, 'RDA: ' + name, fontproperties=font2)
    ax.text(0, 1.89, 'Mode: Precipitation', fontproperties=font2)
    ax.text(0, 1.85, 'Elev: {:.2f}deg'.format(elev), fontproperties=font2)
    if draw_author:
        ax.text(0, 1.73, 'Made by HCl', fontproperties=font2)

def save(folderpath, code, timestr, elev, drange, datatype):
    plt.savefig('{}{}_{}_{:.1f}_{}_{}.png'.format(
        folderpath, code, timestr, elev, drange, datatype.upper()), bbox_inches='tight', pad_inches = 0)
    plt.cla()

def add_shp(m):
    m.readshapefile(modpath + '\\shapefile\\County', 'states', drawbounds=True, linewidth=0.5, color='grey')
    m.readshapefile(modpath + '\\shapefile\\City', 'states', drawbounds=True, linewidth=0.7, color='lightgrey')
    m.readshapefile(modpath + '\\shapefile\\Province', 'states', drawbounds=True, linewidth=1, color='white')

def setup_basemap(lon, lat):
    m = Basemap(llcrnrlon=lon.min(), urcrnrlon=lon.max(), llcrnrlat=lat.min(), urcrnrlat=lat.max(), resolution="l")
    return m

def change_cbar_text(cbar, tick, text):
    cbar.set_ticks(tick)
    cbar.set_ticklabels(text)