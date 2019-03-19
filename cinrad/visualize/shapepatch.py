# -*- coding: utf-8 -*-
# Author: Du puyuan

import os

import numpy as np
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from cinrad.constants import MODULE_DIR
from cinrad.error import RadarPlotError

def highlight_area(area:str, facecolor='None', edgecolor='red', **kwargs):
    r'''Return pathpatch for given area name'''
    fpath = os.path.join(MODULE_DIR, 'shapefile', 'City')
    shp = shapefile.Reader(fpath)
    rec = shp.shapeRecords()
    vertices = list()
    codes = list()
    if isinstance(area, str):
        area = [area]
    for i in area:
        if not isinstance(i, str):
            raise RadarPlotError('Area name should be str')
        name = np.array([i.record[2].decode('GBK') for i in rec])
        target = np.array(rec)[(name == i).nonzero()[0]]
        for j in target:
            codes += [Path.MOVETO] + [Path.LINETO] * (len(j.shape.points) - 1)
            vertices += j.shape.points
        codes += [Path.CLOSEPOLY]
        vertices += [j.shape.points[0]]
        path = Path(vertices, codes)
    patch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    return patch