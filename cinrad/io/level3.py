# -*- coding: utf-8 -*-
# Author: Puyuan Du

from collections import OrderedDict, defaultdict
from typing import Union, Any
import datetime

import numpy as np

from cinrad.projection import get_coordinate
from cinrad.constants import deg2rad
from cinrad._typing import Boardcast_T
from cinrad.io.base import BaseRadar, prepare_file
from cinrad.io._dtype import *
from cinrad.datastruct import Radial, Grid
from cinrad.error import RadarDecodeError

def xy2polar(x:Boardcast_T, y:Boardcast_T) -> tuple:
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y) * 180 / np.pi

class PUP(BaseRadar):
    r'''
    Class handling PUP data (Nexrad Level III data)
    '''
    def __init__(self, file:Any):
        from metpy.io.nexrad import Level3File
        f = Level3File(file)
        self.dtype = self._det_product_type(f.prod_desc.prod_code)
        self.radial_flag = self._is_radial(f.prod_desc.prod_code)
        data_block = f.sym_block[0][0]
        data = np.ma.array(data_block['data'])
        data[data == 0] = np.ma.masked
        self.data = np.ma.masked_invalid(f.map_data(data))
        self.max_range = f.max_range
        if self.radial_flag:
            self.az = np.array(data_block['start_az'] + [data_block['end_az'][-1]]) * deg2rad
            self.rng = np.linspace(0, f.max_range, data.shape[-1] + 1)
        else:
            # TODO: Support grid type data
            raise NotImplementedError('Grid-type data is not supported')
            xdim, ydim = data.shape
            x = np.linspace(xdim * f.ij_to_km * -1, xdim * f.ij_to_km, xdim) / 111 + f.lon
            y = np.linspace(ydim * f.ij_to_km, ydim * f.ij_to_km * -1, ydim) / 111 + f.lat
            self.lon, self.lat = np.meshgrid(x, y)
            self.reso = f.ij_to_km
        self.stationlat = f.lat
        self.stationlon = f.lon
        self.el = f.metadata['el_angle']
        self.scantime = f.metadata['vol_time']
        o = open(file, 'rb')
        o.seek(12)
        code = np.frombuffer(o.read(2), '>i2')[0]
        if code in range(0, 100):
            cds = '0{}'.format(code)
        else:
            cds = str(code)
        self.code = 'Z9' + cds
        o.close()
        self._update_radar_info()

    def get_data(self) -> Union[Grid, Radial]:
        if self.radial_flag:
            lon, lat = self.projection()
            return Radial(self.data, self.max_range, self.el, 1, self.code, self.name, self.scantime,
                          self.dtype, self.stationlon, self.stationlat, lon, lat)
        else:
            return Grid(self.data, self.max_range, self.reso, self.code, self.name, self.scantime,
                        self.dtype, self.lon, self.lat)

    @staticmethod
    def _is_radial(code:int) -> bool:
        return code in range(16, 31)

    def projection(self) -> tuple:
        return get_coordinate(self.rng, self.az, self.el, self.stationlon, self.stationlat, h_offset=False)

    @staticmethod
    def _det_product_type(spec:int) -> str:
        if spec in range(16, 22):
            return 'REF'
        elif spec in range(22, 28):
            return 'VEL'
        elif spec in range(28, 31):
            return 'SW'
        elif spec == 37:
            return 'CR'
        else:
            raise RadarDecodeError('Unsupported product type {}'.format(spec))

class SWAN(object):
    dtype_conv = {0:'B', 1:'b', 2:'u2', 3:'i2', 4:'u2'}
    size_conv = {0:1, 1:1, 2:2, 3:2, 4:2}
    def __init__(self, file:Any):
        f = prepare_file(file)
        header = np.frombuffer(f.read(1024), SWAN_dtype)
        xdim, ydim, zdim = header['x_grid_num'][0], header['y_grid_num'][0], header['z_grid_num'][0]
        dtype = header['m_data_type'][0]
        data_size = int(xdim) * int(ydim) * int(zdim) * self.size_conv[dtype]
        bittype = self.dtype_conv[dtype]
        data_body = np.frombuffer(f.read(data_size), bittype).astype(int)
        if zdim == 1:
            out = data_body.reshape(xdim, ydim)
        else:
            out = data_body.reshape(zdim, xdim, ydim)
        self.data_time = datetime.datetime(header['year'], header['month'], header['day'], header['hour'], header['minute'])
        # TODO: Recognize correct product name
        self.product_name = b''.join(header['data_name'][0, :4]).decode()
        start_lon = header['start_lon'][0]
        start_lat = header['start_lat'][0]
        center_lon = header['center_lon'][0]
        center_lat = header['center_lat'][0]
        end_lon = center_lon * 2 - start_lon
        end_lat = center_lat * 2 - start_lat
        #x_reso = header['x_reso'][0]
        #y_reso = header['y_reso'][0]
        self.lon = np.linspace(start_lon, end_lon, xdim) # For shape compatibility
        self.lat = np.linspace(start_lat, end_lat, ydim)
        if self.product_name == 'CR':
            self.data = np.ma.array((out - 66) / 2, mask=(out==0))
        else:
            self.data = np.ma.array(out, mask=(out==0))

    def get_data(self) -> Grid:
        x, y = np.meshgrid(self.lon, self.lat)
        grid = Grid(self.data, np.nan, np.nan, 'SWAN', 'SWAN', self.data_time, self.product_name, x, y)
        return grid

class StormTrackInfo(object):

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

    def get_all_id(self) -> list:
        return list(self.info.keys())

    def current(self, storm_id:str) -> tuple:
        curpos = self.info[storm_id]['current storm position']
        dist, az = xy2polar(*curpos)
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
        pol_pos = xy2polar(x_pos, y_pos)
        lon = list()
        lat = list()
        for dis, azi in zip(pol_pos[0], pol_pos[1]):
            pos_tup = get_coordinate(dis, azi * deg2rad, 0, self.handler.lon, self.handler.lat, h_offset=False)
            lon.append(pos_tup[0])
            lat.append(pos_tup[1])
        return np.array(lon), np.array(lat)

class HailIndex(object):
    def __init__(self, filepath:str):
        from metpy.io.nexrad import Level3File
        self.handler = Level3File(filepath)
        self.info = self.get_all_hi()
        self.storm_list = self.get_all_id()

    def get_all_hi(self) -> OrderedDict:
        f = self.handler
        if not hasattr(f, 'sym_block'):
            return OrderedDict()
        else:
            data_block = f.sym_block[0]
            sti_data = OrderedDict()
            data_dict = [i for i in data_block if isinstance(i, defaultdict)]
            storm_id = [i for i in data_dict if i['type'] == 'Storm ID']
            info = [i for i in data_dict if i['type'] == 'HDA']
            for sid, inf in zip(storm_id, info):
                stid = sid['id']
                sti_data[stid] = defaultdict()
                sti_data[stid]['current storm position'] = tuple([sid['x'], sid['y']])
                sti_data[stid]['POH'] = inf['POH']
                sti_data[stid]['POSH'] = inf['POSH']
                sti_data[stid]['MESH'] = inf['Max Size']
            return sti_data

    def get_all_id(self) -> list:
        return list(self.info.keys())

    def get_hail_param(self, storm_id:str) -> dict:
        out = dict(self.info[storm_id])
        xy = out.pop('current storm position')
        dist, az = xy2polar(*xy)
        lonlat = get_coordinate(dist, az * deg2rad, 0, self.handler.lon, self.handler.lat, h_offset=False)
        out['position'] = tuple(lonlat)
        return out