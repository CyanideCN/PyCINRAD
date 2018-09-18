# -*- coding: utf-8 -*-
#Author: Du puyuan

import numpy as np
from scipy.interpolate import griddata

def resample(data, distance, azimuth, xreso, yreso):
    #Target grid
    Rrange = np.arange(self.Rreso, drange + self.Rreso, self.Rreso)
    Trange = np.arange(0, 361, 1)
    dist, theta = np.meshgrid(Rrange, Trange)
    #Original grid
    d, t = np.meshgrid(distance, azimuth)
    r = griddata((d.flatten(), t.flatten()), data.flatten(), (dist, theta), method='nearest')
    return r

def grid_2d(data, x, y, resolution=(500, 500)):
    r_x, r_y = resolution
    x_cor = np.linspace(x.min(), x.max(), r_x)
    y_cor = np.linspace(y.min(), y.max(), r_y)
    t_x, t_y = np.meshgrid(x_cor, y_cor)
    r = griddata((x.flatten(), y.flatten()), data.flatten(), (t_x, t_y), method='nearest')
    return r, x_cor, y_cor
