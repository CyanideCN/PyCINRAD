# -*- coding: utf-8 -*-
# Author: Du puyuan

import os

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from cartopy.io import shapereader
import cartopy.crs as ccrs

from cinrad.constants import font, MODULE_DIR
from cinrad.visualize.shapepatch import highlight_area
import xlrd  # for reading city names

def setup_plot(dpi, figsize=(10, 10), style='black'):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    if style == 'black':
        plt.style.use('dark_background')
    return fig

def setup_axes(fig, cmap, norm):
    ax = fig.add_axes([0.92, 0.12, 0.04, 0.35])
    cbar = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical', drawedges=False)
    cbar.ax.tick_params(axis='both', which='both', length=0, labelsize=8)
    cbar.outline.set_visible(False)
    return ax, cbar

def text(ax, drange, reso, scantime, name, elev):
    ax.text(0, 2.09, 'Range: {:.0f}km'.format(drange), fontproperties=font)
    ax.text(0, 2.05, 'Resolution: {:.2f}km'.format(reso) , fontproperties=font)
    ax.text(0, 2.01, 'Date: {}'.format(scantime.strftime('%Y.%m.%d')), fontproperties=font)
    ax.text(0, 1.97, 'Time: {}'.format(scantime.strftime('%H:%M')), fontproperties=font)
    if name is None:
        name = 'Unknown'
    ax.text(0, 1.93, 'RDA: ' + name, fontproperties=font)
    ax.text(0, 1.89, 'Mode: Precipitation', fontproperties=font)
    ax.text(0, 1.85, 'Elev: {:.2f}deg'.format(elev), fontproperties=font)

def save(fpath):
    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
    plt.close('all')

def add_shp(ax, coastline=False, style='black', minlon=None, maxlon=None, minlat=None, maxlat=None, add_city_names=False):
    root = os.path.join(MODULE_DIR, 'shapefile')
    flist = [os.path.join(root, i) for i in ['County', 'City', 'Province']]
    shps = [shapereader.Reader(i).geometries() for i in flist]
    if style == 'black':
        line_colors = ['grey', 'lightgrey', 'white']
    elif style == 'white':
        line_colors = ['lightgrey', 'grey', 'black']
    ax.add_geometries(shps[0], ccrs.PlateCarree(), edgecolor=line_colors[0], facecolor='None', zorder=1, linewidth=0.5)
    ax.add_geometries(shps[1], ccrs.PlateCarree(), edgecolor=line_colors[1], facecolor='None', zorder=1, linewidth=0.7)
    ax.add_geometries(shps[2], ccrs.PlateCarree(), edgecolor=line_colors[2], facecolor='None', zorder=1, linewidth=1)
    if coastline:
        ax.coastlines(resolution='10m', color=line_colors[2], zorder=1, linewidth=1)

    if (add_city_names): #增加判断，是否要绘制城市名称
        # add city names --- Modified By Fulang WU at 2019-01-30
        stationList = os.path.join(root, 'StationNames.xlsx')
        wb = xlrd.open_workbook(stationList) #打开文件
        sheet1 = wb.sheet_by_index(0) #通过索引获取表格
        citynames = sheet1.col_values(0) #获取列内容
        lat = sheet1.col_values(1) #获取列内容
        lon = sheet1.col_values(2) #获取列内容
        for i in range(1,len(citynames)):
            if (lon[i]>=minlon) and (lon[i]<=maxlon) and (lat[i]>=minlat) and (lat[i]<=maxlat): #城市经纬度是否在绘图区域范围内
                ax.text(lon[i], lat[i], r'.', transform=ccrs.PlateCarree(), size=30, color='white') #描点
                ax.text(lon[i]-0.08, lat[i]-0.08, citynames[i], transform=ccrs.PlateCarree(), size=16, color='DarkGray') #标注城市名称

def change_cbar_text(cbar, tick, text):
    cbar.set_ticks(tick)
    cbar.set_ticklabels(text)

def draw_highlight_area(area):
    patch = highlight_area(area)
    ax_ = plt.gca()
    pat = ax_.add_patch(patch)
    pat.set_zorder(2)

def set_geoaxes(lon, lat, extent=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.background_patch.set_fill(False)
    if not extent:
        x_min, x_max, y_min, y_max = lon.min(), lon.max(), lat.min(), lat.max()
    else:
        x_min, x_max, y_min, y_max = extent[0], extent[1], extent[2], extent[3]
    ax.set_extent([x_min, x_max, y_min, y_max], ccrs.PlateCarree())
    return ax