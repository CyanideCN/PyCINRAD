# -*- coding: utf-8 -*-
# Author: Du puyuan

import os

import numpy as np
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from cinrad.constants import modpath
from cinrad.error import RadarPlotError

def highlight_area(area, facecolor='None', edgecolor='red', **kwargs):
    r'''Return pathpatch for given area name'''
    fpath = os.path.join(modpath, 'shapefile', 'City')
    shp = shapefile.Reader(fpath)
    rec = shp.shapeRecords()
    vertices = list()
    codes = list()
    for i in area:
        if not isinstance(i, str):
            raise RadarPlotError('Area name should be str')
        name = np.array([i.record[2].decode('GBK') for i in rec])
        mask = np.ma.array(name, mask=(name==i))
        target = np.array(rec)[mask.mask]
        for j in target:
            codes += [Path.MOVETO] + [Path.LINETO] * (len(j.shape.points) - 1)
            vertices += j.shape.points
        codes += [Path.CLOSEPOLY]
        vertices += [j.shape.points[0]]
        path = Path(vertices, codes)
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    return patch