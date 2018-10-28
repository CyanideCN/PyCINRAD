# -*- coding: utf-8 -*-
# Author: Du puyuan

from ..constants import rhi_cmap_smooth, norm1

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

class RHI:
    def __init__(self, data, hlim=15):
        self.data = data
        self.hlim = hlim
        self.path_customize = False

    def _plot(self, fpath):
        rhi = self.data.data
        xcor = self.data.xcor
        ycor = self.data.ycor
        rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4), dpi=200)
        plt.tricontourf(xcor.flatten(), ycor.flatten(), rhi.flatten(), 128, cmap=rhi_cmap_smooth
                        , norm=norm1)
        plt.ylim(0, self.hlim)
        plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                    data.name, data.az, data.tstr[:4], data.tstr[4:6], data.tstr[6:8], data.tstr[8:10]
                    , data.tstr[10:12], rmax))
        plt.ylabel('Altitude (km)')
        plt.xlabel('Range (km)')
        if self.path_customize:
            path_string = fpath
        else:
            path_string = '{}{}_{}_RHI_{}_{}.png'.format(fpath, self.data.code, self.data.tstr, self.data.drange, self.data.az)
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