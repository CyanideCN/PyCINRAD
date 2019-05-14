# -*- coding: utf-8 -*-
# Author: Puyuan Du

import abc
import os
import pickle
from typing import Optional

import numpy as np

from cinrad.constants import MODULE_DIR
from cinrad.error import RadarDecodeError
from cinrad._typing import number_type

with open(os.path.join(MODULE_DIR, 'data', 'RadarStation.pickle'), 'rb') as buf:
    radarinfo = pickle.load(buf)

def _get_radar_info(code:Optional[str]) -> tuple:
    r'''Get radar station info from the station database according to the station code.'''
    if code is None:
        return ('None', 0, 0, '', 0)
    try:
        return radarinfo[code]
    except IndexError:
        raise RadarDecodeError('Invalid radar code {}'.format(code))

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