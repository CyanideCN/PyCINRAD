# -*- coding: utf-8 -*-
# Author: Puyuan Du

class RadarDecodeError(Exception):
    r'''Unable to decode radar files correctly'''
    pass

class RadarPlotError(Exception):
    r'''Unable to generate visualization of radar data'''
    pass

class RadarCalculationError(Exception):
    r'''Unable to calculate derivatives of radar data'''
    pass