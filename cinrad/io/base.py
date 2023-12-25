# -*- coding: utf-8 -*-
# Author: Puyuan Du

import abc
import os
import json
from typing import Optional, Any
import bz2
import gzip

import numpy as np

from cinrad.constants import MODULE_DIR
from cinrad._typing import Number_T

with open(os.path.join(MODULE_DIR, "data", "radar_station.json"), "r") as buf:
    radarinfo = json.load(buf)


def _get_radar_info(code: Optional[str]) -> tuple:
    r"""Get radar station info from the station database according to the station code."""
    try:
        return radarinfo[code]
    except KeyError:
        return ("None", 0, 0, "", 0)


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


class RadarBase(abc.ABC):
    r"""
    Base class for readers in `cinrad.io`.
    Only used when subclassed
    """

    # Same methods for all radar classes
    def _update_radar_info(self):
        r"""Update radar station info automatically."""
        info = _get_radar_info(self.code)
        self.stationlon = info[1]
        self.stationlat = info[2]
        self.name = info[0]
        self.radarheight = info[4]

    def set_code(self, code: str):
        self.code = code
        self._update_radar_info()

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
