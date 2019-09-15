# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os

__all__ = ["deg2rad", "rm", "con", "con2", "MODULE_DIR"]

MODULE_DIR = os.path.dirname(__file__)
deg2rad = 3.141592653589793 / 180
rm = 8500
con = (180 / 4096) * 0.125
con2 = 0.001824  # calculated manually
vil_const = 3.44e-6
