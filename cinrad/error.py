# -*- coding: utf-8 -*-
# Author: Puyuan Du


class RadarDecodeError(Exception):
    r"""Unable to decode radar files correctly"""
    pass


class RadarPlotError(Exception):
    r"""Unable to generate visualization of radar data"""
    pass


class RadarCalculationError(Exception):
    r"""Unable to calculate derivatives of radar data"""
    pass


class ExceptionOnCall(object):
    r"""Raise exception when calling"""

    def __init__(self, exec_: Exception, msg: str):
        self.exec = exec_
        self.msg = msg

    def __call__(self, *args, **kwargs):
        raise self.exec(self.msg)
