# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from cinrad.datastruct import Slice_
from cinrad._element import sec_plot, norm_plot, prodname

__all__ = ['Section', 'RHI']

class Section:
    def __init__(self, data:Slice_, hlim:int=15):
        self.data = data
        self.dtype = data.dtype
        self.hlim = hlim
        self.path_customize = False

    def _plot(self, fpath:str):
        from cinrad.constants import plot_kw
        rhi = self.data.data
        xcor = self.data.xcor
        ycor = self.data.ycor
        rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
        plt.style.use('dark_background')
        #plt.figure(figsize=(10, 4), dpi=200)
        plt.figure(figsize=(10, 8), dpi=600)  ## 修改于2019-01-22 By WU Fulang
        plt.tick_params(labelsize=20) #坐标轴字体大小
        plt.grid(True, linewidth=0.50, linestyle="-.", color='white') ## 修改于2019-01-22 By WU Fulang
        #plt.contourf(xcor, ycor, rhi, 128, cmap=rhi_cmap_smooth, norm=norm1)
        plt.contourf(xcor, ycor, rhi, 128, cmap=sec_plot[self.data.dtype], norm=norm_plot[self.data.dtype])
        plt.ylim(0, self.hlim)
        stps = self.data.geoinfo['stp_s']
        enps = self.data.geoinfo['enp_s']
        stp = self.data.geoinfo['stp']
        enp = self.data.geoinfo['enp']
        plt.title('Vertical cross-section ({})\nStation: {} Start: {} End: {} Time: {} Max: {}'.format(
                  prodname[self.dtype],self.data.name, stps, enps, self.data.scantime.strftime('%Y.%m.%d %H:%M'), rmax),
                  **plot_kw)
        #重新绘制VCS的横坐标，分为5等分
        deltaLat = (enp[1]-stp[1])/5.0
        deltaLon = (enp[0]-stp[0])/5.0
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['{:.2f}N\n{:.2f}E'.format(stp[1], stp[0]), '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*1., stp[0]+deltaLon*1.),
                                    '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*2., stp[0]+deltaLon*2.), '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*3., stp[0]+deltaLon*3.),
                                '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*4., stp[0]+deltaLon*4.), '{:.2f}N\n{:.2f}E'.format(enp[1], enp[0])]) #分为五等分
        plt.ylabel('Height (km)', **plot_kw, fontsize=23) ## 修改于2019-01-22 By WU Fulang
            #plt.xticks([0, 1], ['{}N\n{}E'.format(stp[1], stp[0]), '{}N\n{}E'.format(enp[1], enp[0])])
        #plt.ylabel('Altitude (km)')
        if self.path_customize:
            path_string = fpath
        else:
            path_string = '{}{}_{}_VCS_{}N{}E_{}N{}E.png'.format(fpath, self.data.code, self.data.scantime.strftime('%Y%m%d%H%M%S'),
                                                                    stp[1], stp[0], enp[1], enp[0])
        plt.savefig(path_string , bbox_inches='tight')

    def __call__(self, *fpath):
        if not fpath:
            fpath = os.path.join(str(Path.home()), 'PyCINRAD')
        else:
            fpath = fpath[0]
            if fpath.upper().endswith('.PNG'):
                self.path_customize = True
            else:
                if not fpath.endswith(os.path.sep):
                    fpath += os.path.sep
        return self._plot(fpath)

class RHI:
    def __init__(self, data:Slice_, hlim:int=15):
        self.data = data
        self.dtype = data.dtype
        self.hlim = hlim
        self.azimuth = data.geoinfo['azimuth']
        self.path_customize = False

    def _plot(self, fpath:str):
        from cinrad.constants import plot_kw
        rhi = self.data.data
        xcor = self.data.xcor
        ycor = self.data.ycor
        rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 5), dpi=300)
        plt.grid(True, linewidth=0.5, linestyle="-.", color='white')
        plt.contourf(xcor, ycor, rhi, 128, cmap=sec_plot[self.data.dtype], norm=norm_plot[self.data.dtype])
        plt.ylim(0, self.hlim)
        plt.title('Range-Height Indicator\nStation: {} Data: {} Range: {:.0f}km Azimuth: {:.0f}° Time: {}'.format(
                  self.data.name, self.data.dtype, self.data.xcor.max(), self.azimuth,
                  self.data.scantime.strftime('%Y-%m-%d %H:%M:%S')))
        plt.ylabel('Height (km)', **plot_kw)
        if self.path_customize:
            path_string = fpath
        else:
            path_string = '{}{}_{}_RHI_{:.0f}_{:.0f}_{}.png'.format(fpath, self.data.code, self.data.scantime.strftime('%Y%m%d%H%M%S'),
                                                            self.azimuth, self.data.xcor.max(), self.dtype)
        plt.savefig(path_string , bbox_inches='tight')

    def __call__(self, *fpath):
        if not fpath:
            fpath = os.path.join(str(Path.home()), 'PyCINRAD')
        else:
            fpath = fpath[0]
            if fpath.upper().endswith('.PNG'):
                self.path_customize = True
            else:
                if not fpath.endswith(os.path.sep):
                    fpath += os.path.sep
        return self._plot(fpath)