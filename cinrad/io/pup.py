# -*- coding: utf-8 -*-
# Author: Puyuan Du

from collections import OrderedDict, defaultdict
from typing import Union, Any

import numpy as np

from cinrad.projection import get_coordinate
from cinrad.constants import deg2rad
from cinrad._typing import boardcast_type

class _StormTrackInfo(object):

    def __init__(self, filepath:str):
        from metpy.io.nexrad import Level3File
        self.handler = Level3File(filepath)
        self.info = self.get_all_sti()
        self.storm_list = self.get_all_id()

    def get_all_sti(self) -> OrderedDict:
        f = self.handler
        if not hasattr(f, 'sym_block'):
            return OrderedDict()
        else:
            data_block = f.sym_block[0]
            sti_data = OrderedDict()
            data_dict = [i for i in data_block if isinstance(i, defaultdict)]
            for i in data_dict:
                if i['type'] == 'Storm ID':
                    sti_data[i['id']] = defaultdict()
                    sti_data[i['id']]['current storm position'] = tuple([i['x'], i['y']])
                else:
                    stid = list(sti_data.keys())[-1]
                    if 'markers' in i.keys():
                        pos = i['markers']
                        if isinstance(pos, dict):
                            pos = [pos]
                        name = list(pos[0].keys())[0]
                        sti_data[stid][name] = list()
                        sti_data[stid][name] += i.get('track')
                    elif 'STI Circle' in i.keys():
                        circle_dict = i['STI Circle']
                        sti_data[stid]['radius'] = circle_dict['radius']
                        sti_data[stid]['current storm position'] = tuple([circle_dict['x'], circle_dict['y']])
        return sti_data

    @staticmethod
    def xy2polar(x:boardcast_type, y:boardcast_type) -> tuple:
        return np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y) * 180 / np.pi

    def get_all_id(self) -> list:
        return list(self.info.keys())

    def current(self, storm_id:str) -> tuple:
        curpos = self.info[storm_id]['current storm position']
        dist, az = self.xy2polar(*curpos)
        lonlat = get_coordinate(dist, az * deg2rad, 0, self.handler.lon, self.handler.lat, h_offset=False)
        return lonlat

    def track(self, storm_id:str, tracktype:str) -> Union[tuple, Any]:
        if tracktype == 'forecast':
            key = 'forecast storm position'
        elif tracktype == 'past':
            key = 'past storm position'
        else:
            raise KeyError('Key {} does not exist'.format(key))
        if key not in self.info[storm_id].keys():
            return None
        forpos = self.info[storm_id][key]
        if forpos == None:
            return
        x_pos = np.array([i[0] for i in forpos])
        y_pos = np.array([i[1] for i in forpos])
        pol_pos = self.xy2polar(x_pos, y_pos)
        lon = list()
        lat = list()
        for dis, azi in zip(pol_pos[0], pol_pos[1]):
            pos_tup = get_coordinate(dis, azi * deg2rad, 0, self.handler.lon, self.handler.lat, h_offset=False)
            lon.append(pos_tup[0])
            lat.append(pos_tup[1])
        return np.array(lon), np.array(lat)
