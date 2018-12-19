# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os

import matplotlib.colors as cmx
from matplotlib.font_manager import FontProperties

from .gpf import _cmap

deg2rad = 3.141592653589793 / 180
rm = 8500
con = (180 / 4096) * 0.125
con2 = 0.001824  # calculated manually

modpath = os.path.dirname(__file__)
CMAP_DIR = os.path.join(modpath, 'colormap')
r_cmap = _cmap('REF')['cmap']
v_cmap = _cmap('VEL')['cmap']
v_cbar = _cmap('VEL_reverse')['cmap']
rhi_cmap_smooth = _cmap('REF_s')['cmap']
r_cmap_smooth = _cmap('REF_s')['cmap']
zdr_cmap = _cmap('ZDR')['cmap']
zdr_cbar = _cmap('ZDR_reverse')['cmap']
kdp_cmap = _cmap('KDP')['cmap']
kdp_cbar = _cmap('KDP_reverse')['cmap']
cc_cmap = _cmap('CC')['cmap']
cc_cbar = _cmap('CC_reverse')['cmap']
et_cmap = _cmap('ET')['cmap']
et_cbar = _cmap('ET_reverse')['cmap']
vil_cmap = _cmap('VIL')['cmap']
vil_cbar = _cmap('VIL_reverse')['cmap']
rf_cmap = cmx.ListedColormap('#660066', '#FFFFFF')
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)
norm3 = cmx.Normalize(-1, 0)
norm4 = cmx.Normalize(0, 1)
norm5 = cmx.Normalize(0, 21)
norm6 = cmx.Normalize(-4, 5)
norm7 = cmx.Normalize(260, 360)
norm8 = cmx.Normalize(0, 0.99)