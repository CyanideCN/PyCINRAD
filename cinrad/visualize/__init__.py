# -*- coding: utf-8 -*-
# Author: Puyuan Du

import matplotlib

if "Arial" in matplotlib.rc_params()["font.sans-serif"]:
    matplotlib.rc("font", family="Arial")
from cinrad.visualize.ppi import *
from cinrad.visualize.rhi import *
