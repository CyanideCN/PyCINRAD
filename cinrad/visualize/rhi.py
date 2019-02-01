# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..constants import rhi_cmap_smooth, norm1, font2

__all__ = ['Section']

class Section:
    def __init__(self, data, hlim=15):
        self.data = data
        self.dtype = data.dtype
        self.hlim = hlim
        self.path_customize = False

    def _plot(self, fpath):
        rhi = self.data.data
        xcor = self.data.xcor
        ycor = self.data.ycor
        rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
        plt.style.use('dark_background')
        #plt.figure(figsize=(10, 4), dpi=200)
        plt.figure(figsize=(10, 8), dpi=600)  ## 修改于2019-01-22 By WU Fulang
        plt.contourf(xcor, ycor, rhi, 128, cmap=rhi_cmap_smooth, norm=norm1)
        plt.ylim(0, self.hlim)
        plt.tick_params(labelsize=20) #坐标轴字体大小
        plt.grid(True, linewidth=0.50, linestyle="-.", color='white') ## 修改于2019-01-22 By WU Fulang
        
        #设置横纵坐标名称对应字体格式
        font1 = {'family':'Times New Roman','weight':'normal','size':23,}
        	
        if self.dtype == 'RHI':
            az = self.data.geoinfo['azimuth']
            plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                      self.data.name, az, self.data.tstr[:4], self.data.tstr[4:6], self.data.tstr[6:8],
                      self.data.tstr[8:10], self.data.tstr[10:12], rmax), fontproperties=font2)
            plt.xlabel('Range (km)',font1) ## 修改于2019-01-22 By WU Fulang
        elif self.dtype == 'VCS':
            stps = self.data.geoinfo['stp_s']
            enps = self.data.geoinfo['enp_s']
            stp = self.data.geoinfo['stp']
            enp = self.data.geoinfo['enp']
            plt.title('Vertical cross-section\nStation: {} Start: {} End: {} Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                      self.data.name, stps, enps, self.data.tstr[:4], self.data.tstr[4:6], self.data.tstr[6:8], self.data.tstr[8:10],
                      self.data.tstr[10:12], rmax), fontproperties=font2)
                      	
            #重新绘制VCS的横坐标，分为5等分
            deltaLat = (enp[1]-stp[1])/5.0
            deltaLon = (enp[0]-stp[0])/5.0
            plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['{:.2f}N\n{:.2f}E'.format(stp[1], stp[0]), '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*1., stp[0]+deltaLon*1.),
            						 '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*2., stp[0]+deltaLon*2.), '{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*3., stp[0]+deltaLon*3.),
            						'{:.2f}N\n{:.2f}E'.format(stp[1]+deltaLat*4., stp[0]+deltaLon*4.), '{:.2f}N\n{:.2f}E'.format(enp[1], enp[0])]) #分为五等分
        plt.ylabel('Height (km)',font1) ## 修改于2019-01-22 By WU Fulang
        if self.path_customize:
            path_string = fpath
        else:
            if self.dtype == 'RHI':
                path_string = '{}{}_{}_RHI_{}_{}.png'.format(fpath, self.data.code, self.data.tstr, self.data.drange, self.data.az)
            elif self.dtype == 'VCS':
                path_string = '{}{}_{}_VCS_{}N{}E_{}N{}E.png'.format(fpath, self.data.code, self.data.tstr, stp[1],
                                                                     stp[0], enp[1], enp[0])
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
