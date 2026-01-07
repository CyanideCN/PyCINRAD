# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

from typing import Any
import bz2
import gzip

import numpy as np

from cinrad._typing import Number_T


def prepare_file(file: Any) -> Any:
    if hasattr(file, "read"):
        return file
    f = open(file, "rb")
    magic = f.read(3)
    f.close()
    if magic.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(file, "rb")
    if magic.startswith(b"BZh"):
        return bz2.BZ2File(file, "rb")
    return open(file, "rb")


class RadarBase(object):
    r"""
    Base class for readers in `cinrad.io`.
    Only used when subclassed
    """

    def __init__(self, *args, **kwargs):
        self.name = ''
        self.stationlat = 0.
        self.stationlon = 0.
        self.radarheight = 0.

    def get_nscans(self) -> int:
        return len(self.el)

    def available_product(self, tilt: int) -> list:
        r"""Get all available products in given tilt"""
        return list(self.data[tilt].keys())

    @staticmethod
    def get_range(drange: Number_T, reso: Number_T) -> np.ndarray:
        rng = np.arange(reso, drange + reso, reso)
        valid_entry = int(drange // reso)
        return rng[:valid_entry]
    
    @staticmethod
    def get_range_safe(start: Number_T, stop: Number_T, step: Number_T) -> np.ndarray:
        r"""
        Level2中切片时使用了//, 所以只能使用上面的get_range, 否则dist长度不一致;
        Level3中没有切片功能, 所以使用这个函数来确保不出现浮点数溢出的长度BUG.
        """
        int_start = int(start / step) + 1
        int_stop = int(stop / step) + 1
        return step * np.arange(int_start, int_stop)

    def decode(self, arg, encoding="ascii") -> str:
        return arg.decode(encoding, "ignore").replace("\x00", "")
    