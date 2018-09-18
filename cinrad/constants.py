# -*- coding: utf-8 -*-
#Author: Du puyuan

from .form_colormap import form_colormap

import os

import matplotlib.colors as cmx
from matplotlib.font_manager import FontProperties

deg2rad = 3.141592653589793 / 180
Rm1 = 8500
con = (180 / 4096) * 0.125
con2 = 0.001824 # calculated manually

folderpath = 'D:\\'

r_cmap = form_colormap('colormap\\r_main.txt', sep=True)
v_cmap = form_colormap('colormap\\v_main.txt', sep=False)
vel_cbar = form_colormap('colormap\\v_cbar.txt', sep=True)
rhi_cmap_smooth = form_colormap('colormap\\r_main.txt', sep=False, spacing='v')
r_cmap_smooth = form_colormap('colormap\\r_smooth.txt', sep=False, spacing='v')
zdr_cmap = form_colormap('colormap\\zdr_main.txt', sep=False)
zdr_cbar = form_colormap('colormap\\zdr_cbar.txt', sep=True)
kdp_cmap = form_colormap('colormap\\kdp_main.txt', sep=False)
kdp_cbar = form_colormap('colormap\\kdp_cbar.txt', sep=True)
cc_cmap = form_colormap('colormap\\cc_main.txt', sep=False)
cc_cbar = form_colormap('colormap\\cc_cbar.txt', sep=True)
et_cmap = form_colormap('colormap\\et_main.txt', sep=False)
et_cbar = form_colormap('colormap\\et_cbar.txt', sep=True)
vil_cmap = form_colormap('colormap\\vil_main.txt', sep=True)
vil_cbar = form_colormap('colormap\\vil_cbar.txt', sep=True)
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)