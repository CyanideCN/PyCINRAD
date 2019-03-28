# -*- coding: utf-8 -*-
# Author: Du puyuan

import os
from typing import Union, List

import numpy as np
import shapefile
from matplotlib.lines import Line2D

from cinrad.constants import MODULE_DIR
from cinrad.error import RadarPlotError
from cinrad._typing import GList

def highlight_area(area:Union[GList, str], linecolor:str='red', **kwargs) -> List[Line2D]:
    r'''Return list of Line2D object for given area name'''
    fpath = os.path.join(MODULE_DIR, 'shapefile', 'City')
    shp = shapefile.Reader(fpath)
    rec = shp.shapeRecords()
    lines = list()
    if isinstance(area, str):
        area = [area]
    for i in area:
        if not isinstance(i, str):
            raise RadarPlotError('Area name should be str')
        name = np.array([i.record[2].decode('GBK') for i in rec])
        target = np.array(rec)[(name == i).nonzero()[0]]
        for j in target:
            pts = j.shape.points
            x = [i[0] for i in pts]
            y = [i[1] for i in pts]
            lines.append(Line2D(x, y, color=linecolor))
    return lines