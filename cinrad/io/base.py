# -*- coding: utf-8 -*-
# Author: Puyuan Du

import abc
import os

import numpy as np

from cinrad.constants import MODULE_DIR
from cinrad.error import RadarDecodeError
from cinrad._typing import number_type

radarinfo = np.load(os.path.join(MODULE_DIR, 'data', 'RadarStation.npy'))

def _get_radar_info(code:str) -> tuple:
    r'''Get radar station info from the station database according to the station code.'''
    try:
        pos = np.where(radarinfo[0] == code)[0][0]
    except IndexError:
        raise RadarDecodeError('Invalid radar code {}'.format(code))
    name = radarinfo[1][pos]
    lon = radarinfo[2][pos]
    lat = radarinfo[3][pos]
    radartype = radarinfo[4][pos]
    radarheight = radarinfo[5][pos]
    return name, lon, lat, radartype, radarheight

class BaseRadar(abc.ABC):
    r'''
    Base class for readers in `cinrad.io`.
    Only used when subclassed
    '''

    # Same methods for all radar classes
    def _update_radar_info(self):
        r'''Update radar station info automatically.'''
        info = _get_radar_info(self.code)
        if info is None:
            warnings.warn('Auto fill radar station info failed, please set code manually', UserWarning)
        else:
            self.stationlon = info[1]
            self.stationlat = info[2]
            self.name = info[0]
            self.radarheight = info[4]

    def set_code(self, code:str):
        self.code = code
        self._update_radar_info()

    def get_nscans(self) -> int:
        return len(self.el)

    def avaliable_product(self, tilt:int) -> list:
        r'''Get all avaliable products in given tilt'''
        return list(self.data[tilt].keys())

    @staticmethod
    def get_range(drange:number_type, reso:number_type) -> np.ndarray:
        return np.arange(reso, drange + reso, reso)