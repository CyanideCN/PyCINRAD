# -*- coding: utf-8 -*-
# Author: Puyuan Du

from .form_colormap import form_colormap

import matplotlib.colors as cmx
from matplotlib.font_manager import FontProperties

import os

deg2rad = 3.141592653589793 / 180
Rm1 = 8500
con = (180 / 4096) * 0.125
con2 = 0.001824  # calculated manually

modpath = os.path.dirname(__file__)
CMAP_DIR = os.path.join(modpath, 'colormap')
r_cmap = form_colormap(os.path.join(CMAP_DIR, 'r_main.txt'), sep=True)
v_cmap = form_colormap(os.path.join(CMAP_DIR, 'v_main.txt'), sep=False)
v_cbar = form_colormap(os.path.join(CMAP_DIR, 'v_cbar.txt'), sep=True)
rhi_cmap_smooth = form_colormap(os.path.join(CMAP_DIR, 'r_smooth.txt'), sep=False, spacing='v')
r_cmap_smooth = form_colormap(os.path.join(CMAP_DIR, 'r_smooth.txt'), sep=False, spacing='v')
zdr_cmap = form_colormap(os.path.join(CMAP_DIR, 'zdr_main.txt'), sep=False)
zdr_cbar = form_colormap(os.path.join(CMAP_DIR, 'zdr_cbar.txt'), sep=True)
kdp_cmap = form_colormap(os.path.join(CMAP_DIR, 'kdp_main.txt'), sep=False)
kdp_cbar = form_colormap(os.path.join(CMAP_DIR, 'kdp_cbar.txt'), sep=True)
cc_cmap = form_colormap(os.path.join(CMAP_DIR, 'cc_main.txt'), sep=False)
cc_cbar = form_colormap(os.path.join(CMAP_DIR, 'cc_cbar.txt'), sep=True)
et_cmap = form_colormap(os.path.join(CMAP_DIR, 'et_main.txt'), sep=False)
et_cbar = form_colormap(os.path.join(CMAP_DIR, 'et_cbar.txt'), sep=True)
vil_cmap = form_colormap(os.path.join(CMAP_DIR, 'vil_main.txt'), sep=True)
vil_cbar = form_colormap(os.path.join(CMAP_DIR, 'vil_cbar.txt'), sep=True)
rf_cmap = cmx.ListedColormap('#660066', '#FFFFFF')
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)
norm3 = cmx.Normalize(-1, 0)
norm4 = cmx.Normalize(0, 1)
norm5 = cmx.Normalize(0, 21)
norm6 = cmx.Normalize(-4, 5)
norm7 = cmx.Normalize(260, 360)
norm8 = cmx.Normalize(0, 0.99)