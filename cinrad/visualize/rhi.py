# -*- coding: utf-8 -*-
# Author: Du puyuan

from ..constants import rhi_cmap_smooth, norm1, font2, folderpath

import os

import matplotlib.pyplot as plt
import numpy as np

def rhi(data, height=15):
    global folderpath
    rhi = data.data
    xcor = data.xcor
    ycor = data.ycor
    rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 4), dpi=200)
    plt.tricontourf(xcor.flatten(), ycor.flatten(), rhi.flatten(), 128, cmap=rhi_cmap_smooth
                    , norm=norm1)
    plt.ylim(0, height)
    plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                data.name, data.az, data.tstr[:4], data.tstr[4:6], data.tstr[6:8], data.tstr[8:10]
                , data.tstr[10:12], rmax), fontproperties=font2)
    plt.ylabel('Altitude (km)')
    plt.xlabel('Range (km)')
    if not folderpath.endswith(os.path.sep):
        folderpath += os.path.sep
    plt.savefig('{}{}_{}_RHI_{}_{}.png'.format(folderpath, data.code, data.tstr, data.drange, data.az)
                , bbox_inches='tight')
