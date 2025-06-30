# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import matplotlib

if "Arial" in matplotlib.rc_params()["font.sans-serif"]:
    matplotlib.rc("font", family="Arial")
from cinrad.visualize.ppi import *
from cinrad.visualize.rhi import *
