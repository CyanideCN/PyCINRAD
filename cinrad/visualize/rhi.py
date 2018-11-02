# -*- coding: utf-8 -*-
# Author: Puyuan Du

from ..constants import rhi_cmap_smooth, norm1, font2

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
        plt.figure(figsize=(10, 4), dpi=200)
        plt.contourf(xcor, ycor, rhi, 128, cmap=rhi_cmap_smooth, norm=norm1)
        plt.ylim(0, self.hlim)
        if self.dtype == 'RHI':
            az = self.data.geoinfo['azimuth']
            plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                      self.data.name, az, self.data.tstr[:4], self.data.tstr[4:6], self.data.tstr[6:8],
                      self.data.tstr[8:10], self.data.tstr[10:12], rmax), fontproperties=font2)
            plt.xlabel('Range (km)')
        elif self.dtype == 'VCS':
            stps = self.data.geoinfo['stp_s']
            enps = self.data.geoinfo['enp_s']
            stp = self.data.geoinfo['stp']
            enp = self.data.geoinfo['enp']
            plt.title('Vertical cross-section\nStation: {} Start: {} End: {} Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                      self.data.name, stps, enps, self.data.tstr[:4], self.data.tstr[4:6], self.data.tstr[6:8], self.data.tstr[8:10],
                      self.data.tstr[10:12], rmax), fontproperties=font2)
            plt.xticks([0, 1], ['{}N\n{}E'.format(stp[1], stp[0]), '{}N\n{}E'.format(enp[1], enp[0])])
        plt.ylabel('Altitude (km)')
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