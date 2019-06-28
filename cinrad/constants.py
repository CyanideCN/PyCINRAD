# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
import warnings

import matplotlib.colors as cmx
from matplotlib.font_manager import FontProperties

from cinrad.gpf import _cmap

__all__ = ['deg2rad', 'rm', 'con', 'con2', 'MODULE_DIR', 'r_cmap', 'v_cmap',
           'v_cbar', 'r_cmap_smooth', 'zdr_cmap', 'zdr_cbar', 'kdp_cmap',
           'kdp_cbar', 'cc_cmap', 'cc_cbar', 'et_cmap', 'et_cbar', 'vil_cmap',
           'vil_cbar', 'rf_cmap', 'norm1', 'norm2', 'norm3', 'norm4',
           'norm5', 'norm6', 'norm7', 'norm8', 'v_cmap_smooth', 'zdr_cmap_smooth',
           'cc_cmap_smooth', 'kdp_cmap_smooth', 'norm9', 'norm10', 'plot_kw']

deg2rad = 3.141592653589793 / 180
rm = 8500
con = (180 / 4096) * 0.125
con2 = 0.001824 # calculated manually
vil_const = 3.44e-6

MODULE_DIR = os.path.dirname(__file__)
CMAP_DIR = os.path.join(MODULE_DIR, 'data', 'colormap')
r_cmap = _cmap('REF')['cmap']
r_cmap_smooth = _cmap('REF_s')['cmap']
v_cmap = _cmap('VEL')['cmap']
v_cbar = _cmap('VEL_reverse')['cmap']
v_cmap_smooth = _cmap('VEL_s')['cmap']
zdr_cmap = _cmap('ZDR')['cmap']
zdr_cbar = _cmap('ZDR_reverse')['cmap']
zdr_cmap_smooth = _cmap('ZDR_s')['cmap']
kdp_cmap = _cmap('KDP')['cmap']
kdp_cbar = _cmap('KDP_reverse')['cmap']
kdp_cmap_smooth = _cmap('KDP_s')['cmap']
cc_cmap = _cmap('CC')['cmap']
cc_cbar = _cmap('CC_reverse')['cmap']
cc_cmap_smooth = _cmap('CC_s')['cmap']
et_cmap = _cmap('ET')['cmap']
et_cbar = _cmap('ET_reverse')['cmap']
vil_cmap = _cmap('VIL')['cmap']
vil_cbar = _cmap('VIL_reverse')['cmap']
rf_cmap = cmx.ListedColormap('#660066', '#FFFFFF')

font = FontProperties(fname=os.path.join(MODULE_DIR, 'data', 'font', 'NotoSansHans-Regular.otf'))
plot_kw = {'fontproperties':font}

norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)
norm3 = cmx.Normalize(-1, 0)
norm4 = cmx.Normalize(0, 1)
norm5 = cmx.Normalize(0, 21)
norm6 = cmx.Normalize(-4, 5)
norm7 = cmx.Normalize(260, 360)
norm8 = cmx.Normalize(0, 0.99)
norm9 = cmx.Normalize(-0.8, 21)
norm10 = cmx.Normalize(0.1, 6)

def set_font(font_path):
    glb = globals()
    font = FontProperties(fname=font_path)
    glb['plot_kw'] = {'fontproperties':font}