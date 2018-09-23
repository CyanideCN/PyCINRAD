#coding=utf-8

from ..constants import modpath

import os

import numpy as np
import shapefile
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt

def highlight_area(area):
    fpath = os.path.join(modpath, 'shapefile\City')
    shp = shapefile.Reader(fpath)
    rec = shp.shapeRecords()

    name = np.array([i.record[2].decode('GBK') for i in rec])
    mask = np.ma.array(name, mask=(name==area))
    target = np.array(rec)[mask.mask]

    vertices = list()
    codes = list()
    for i in target:
        codes += [Path.MOVETO] + [Path.LINETO] * (len(i.shape.points) - 1)
        vertices += i.shape.points
    codes += [Path.CLOSEPOLY]
    vertices += [i.shape.points[0]]
    path = Path(vertices, codes)
    patch = PathPatch(path, facecolor='None', edgecolor='red')
    return patch