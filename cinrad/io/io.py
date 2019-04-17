# -*- coding: utf-8 -*-
# Author: Puyuan Du

import warnings
import datetime
from pathlib import Path
import bz2
import gzip

from collections import namedtuple, defaultdict
from typing import Union, Optional, List, Any

import numpy as np

from cinrad.constants import deg2rad, con, con2
from cinrad.datastruct import Radial, Grid, Slice_
from cinrad.projection import get_coordinate, height
from cinrad.error import RadarDecodeError
from cinrad.io._io import NetCDFWriter
from cinrad.io.base import BaseRadar
from cinrad.io._dtype import *
from cinrad._typing import number_type

__all__ = ['CinradReader', 'StandardData', 'NexradL2Data', 'PUP', 'SWAN']

ScanConfig = namedtuple('ScanConfig', SDD_cut.fields.keys())

def merge_bytes(byte_list:List[bytes]) -> bytes:
    return b''.join(byte_list)

def _detect_radartype(f:Any, filename:str, type_assert:Optional[str]=None) ->tuple:
    r'''Detect radar type from records in file'''
    # Attempt to find information in file, which has higher
    # priority compared with information obtained from file name
    f.seek(100)
    typestring = f.read(9)
    det_sc = typestring == b'CINRAD/SC'
    det_cd = typestring == b'CINRAD/CD'
    f.seek(116)
    det_cc = f.read(9) == b'CINRAD/CC'
    # Read information from filename (if applicable)
    if filename.startswith('RADA'):
        spart = filename.split('-')
        code = spart[1]
        radartype = spart[2]
    elif filename.startswith('Z'):
        spart = filename.split('_')
        code = spart[3]
        radartype = spart[7]
    else:
        radartype = None
        code = None
    if det_sc:
        radartype = 'SC'
    elif det_cd:
        radartype = 'CD'
    elif det_cc:
        radartype = 'CC'
    if type_assert:
        radartype = type_assert
    if radartype is None:
        raise RadarDecodeError('Radar type undefined')
    return code, radartype

def prepare_file(file):
    if hasattr(file, 'read'):
        return file
    f = open(file)
    magic = f.read(3)
    f.close()
    if magic.startswith(b'\x1f\x8b'):
        return gzip.GzipFile(file, 'rb')
    if magic.startswith(b'BZh'):
        return bz2.BZ2File(file, 'rb')
    return open(file, 'rb')

class CinradReader(BaseRadar):
    r'''
    Class handling CINRAD radar reading

    Attributes
    ----------
    radartype: str
        type of radar (SA, SB, etc.)
    scantime: datetime.datetime
        time of scan for this data
    code: str
        code for this radar
    angleindex_r: list
        indices of tilts which have reflectivity data
    angleindex_v: list
        indices of tilts which have velocity data
    stationlon: float
        logitude of this radar
    stationlat: float
        latitude of this radar
    radarheight: float
        height of this radar
    name: str
        name of this radar
    Rreso: float
        radial resolution of reflectivity data
    Vreso: float
        radial resolution of velocity data
    a_reso: int
        number of radials in one scan
    el: np.ndarray
        elevation angles for tilts
    drange: float
        current selected radius of data
    elev: float
        elevation angle of current selected tilt
    tilt: int
        current selected tilt
    '''
    def __init__(self, file:Any, radar_type:Optional[str]=None, file_name:Optional[str]=None):
        r'''
        Parameters
        ----------
        file: str / obj with read method
            path directed to the file to read / file object
        radar_type: str, optional
            type of radar
        '''
        f = prepare_file(file)
        filename = Path(file).name if isinstance(file, str) else ''
        self.code, radartype = _detect_radartype(f, filename, type_assert=radar_type)
        if radar_type:
            if radartype is not radar_type:
                warnings.warn('Contradictory information from input radar type and\
                              radar type detected from input file.')
            self.radartype = radar_type
        else:
            self.radartype = radartype
        f.seek(0)
        if radartype in ['SA', 'SB']:
            self._SAB_handler(f)
        elif radartype in ['CA', 'CB']:
            self._SAB_handler(f, SAB=False)
        elif radartype is 'CC':
            self._CC_handler(f)
        elif radartype is 'SC':
            self._SC_handler(f)
        elif radartype is 'CD':
            raise RadarDecodeError('CD radar is not supported')
        else:
            raise RadarDecodeError('Unrecognized data')
        self._update_radar_info()
        f.close()

    def _SAB_handler(self, f:Any, SAB:bool=True):
        if SAB:
            radar_dtype = SAB_dtype
        else:
            radar_dtype = CAB_dtype
        data = np.frombuffer(f.read(), dtype=radar_dtype)
        start = datetime.datetime(1969, 12, 31)
        deltday = datetime.timedelta(days=int(data['day'][0]))
        deltsec = datetime.timedelta(milliseconds=int(data['time'][0]))
        self.scantime = start + deltday + deltsec
        self.Rreso = data['gate_length_r'][0] / 1000
        self.Vreso = data['gate_length_v'][0] / 1000
        boundary = np.where(data['radial_num']==1)[0]
        self.el = data['elevation'][boundary] * con
        self.azimuth = data['azimuth'] * con * deg2rad
        dv = data['v_reso'][0]
        self.nyquist_v = data['nyquist_vel'][boundary] / 100
        f.seek(0)
        size = radar_dtype.itemsize
        b = np.append(boundary, data.shape[0] - 1)
        gnr = data['gate_num_r'][boundary]
        gnv = data['gate_num_v'][boundary]
        out_data = dict()
        for bidx, rnum, vnum, idx in zip(np.diff(b), gnr, gnv, range(len(b))):
            temp_dtype = [('header', 'u1', 128),
                          ('ref', 'u1', rnum),
                          ('vel', 'u1', vnum),
                          ('sw', 'u1', vnum),
                          ('res', 'u1', size - 128 - rnum - vnum * 2)]
            da = np.frombuffer(f.read(bidx * size), dtype=np.dtype(temp_dtype))
            out_data[idx] = dict()
            r = (np.ma.array(da['ref'], mask=(da['ref'] == 0)) - 2) / 2 - 32
            r[r == 95.5] = 0
            out_data[idx]['REF'] = r
            v = np.ma.array(da['vel'], mask=(da['vel'] < 2))
            sw = np.ma.array(da['sw'], mask=(da['sw'] < 2))
            if dv == 2:
                out_data[idx]['VEL'] = (v - 2) / 2 - 63.5
                out_data[idx]['SW'] = (sw - 2) / 2 - 63.5
            elif dv == 4:
                out_data[idx]['VEL'] = v - 2 - 127
                out_data[idx]['SW'] = sw - 2 - 127
            out_data[idx]['azimuth'] = self.azimuth[b[idx]:b[idx + 1]]
            out_data[idx]['RF'] = np.ma.array(da['vel'], mask=(da['vel'] != 1))
        angleindex = np.arange(0, data['el_num'][-1], 1)
        self.angleindex_r = np.delete(angleindex, [1, 3])
        self.angleindex_v = np.delete(angleindex, [0, 2])
        self.data = out_data

    def _CC_handler(self, f:Any):
        header = np.frombuffer(f.read(1024), CC_header)
        utc_offset = datetime.timedelta(hours=8)
        self.scantime = datetime.datetime(header['ucEYear1'][0] * 100 + header['ucEYear2'][0], header['ucEMonth'][0],
                                          header['ucEDay'][0], header['ucEHour'], header['ucEMinute'],
                                          header['ucESecond']) - utc_offset
        f.seek(218)
        param = np.frombuffer(f.read(660), CC_param)
        stop_angle = np.where(param['usAngle'] < param['usAngle'][0])[0][0]
        self.el = param['usAngle'][:stop_angle] / 100
        f.seek(1024)
        data = np.frombuffer(f.read(), CC_data)
        r = np.ma.array(data['Z'], mask=(data['Z'] == -0x8000)) / 10
        v = np.ma.array(data['V'], mask=(data['V'] == -0x8000)) / 10
        w = np.ma.array(data['W'], mask=(data['W'] == -0x8000)) / 10
        self.Rreso = 0.3
        self.Vreso = 0.3
        data = dict()
        for i in range(len(self.el)):
            data[i] = dict()
            data[i]['REF'] = r[i * 512: (i + 1) * 512]
            data[i]['VEL'] = v[i * 512: (i + 1) * 512]
            data[i]['SW'] = w[i * 512: (i + 1) * 512]
            data[i]['azimuth'] = self.get_azimuth_angles(i)
        self.data = data
        self.angleindex_r = self.angleindex_v = [i for i in range(len(self.el))]

    def _SC_handler(self, f:Any):
        vraw = list()
        rraw = list()
        blocklength = 4000
        utc_offset = datetime.timedelta(hours=8)
        f.seek(853)
        self.scantime = (datetime.datetime(year=np.frombuffer(f.read(2), 'u2')[0], month=np.frombuffer(f.read(1), 'u1')[0],
                                          day=np.frombuffer(f.read(1), 'u1')[0], hour=np.frombuffer(f.read(1), 'u1')[0],
                                          minute=np.frombuffer(f.read(1), 'u1')[0], second=np.frombuffer(f.read(1), 'u1')[0])
                         - utc_offset)
        f.seek(1024)
        self.Rreso = 0.3
        self.Vreso = 0.3
        elev = list()
        count = 0
        while count < 3240:
            q = f.read(blocklength)
            elev.append(np.frombuffer(q[2:4], 'u2')[0])
            x = np.frombuffer(q[8:], 'u1').astype(float)
            rraw.append(x[slice(None, None, 4)])
            vraw.append(x[slice(1, None, 4)])
            count += 1
        rraw = np.concatenate(rraw).reshape(3240, 998)
        vraw = np.concatenate(vraw).reshape(3240, 998)
        r = np.ma.array(rraw, mask=(rraw == 0))
        r1 = (r - 64) / 2
        v = np.ma.array(vraw, mask=(vraw == 0))
        v1 = (v - 128) / 2
        self.el = np.array(elev[slice(359, None, 360)]) * con2
        data = dict()
        for i in range(len(self.el) - 1):
            data[i] = dict()
            data[i]['REF'] = r1[i * 360: (i + 1) * 360]
            data[i]['VEL'] = v1[i * 360: (i + 1) * 360]
            data[i]['azimuth'] = self.get_azimuth_angles(i)
        self.data = data
        self.angleindex_r = self.angleindex_v = list(self.data.keys())

    def get_nrays(self, scan:int) -> int:
        r'''Get number of radials in certain scan'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            return len(self.data[scan]['azimuth'])
        elif self.radartype == 'CC':
            return 512
        elif self.radartype == 'SC':
            return 360

    def get_azimuth_angles(self, scans:Optional[int]=None) -> np.ndarray:
        r'''Get index of input azimuth angle (radian)'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            if scans is None:
                return self.azimuth
            else:
                return self.data[scans]['azimuth']
        elif self.radartype == 'CC':
            if scans is None:
                return np.array(np.linspace(0, 360, 512).tolist() * self.get_nscans()) * deg2rad
            else:
                return np.array(np.linspace(0, 360, 512).tolist()) * deg2rad
        elif self.radartype == 'SC':
            if scans is None:
                return np.array(np.linspace(0, 360, 360).tolist() * self.get_nscans()) * deg2rad
            else:
                return np.array(np.linspace(0, 360, 360).tolist()) * deg2rad

    def get_elevation_angles(self, scans:Optional[int]=None) -> Union[np.ndarray, float]:
        if scans is None:
            return self.el
        else:
            return self.el[scans]

    def get_data(self, tilt:int, drange:number_type, dtype:str) -> Radial:
        r'''
        Get radar raw data

        Parameters
        ----------
        tilt: int
            index of elevation angle
        drange: float
            radius of data
        dtype: str
            type of product (REF, VEL, etc.)

        Returns
        -------
        r_obj: cinrad.datastruct.Radial
        '''
        rf_flag = False
        self.tilt = tilt
        reso = self.Rreso if dtype == 'REF' else self.Vreso
        dmax = np.round(self.data[tilt][dtype][0].shape[0] * reso)
        if dmax < drange:
            warnings.warn('Requested data range exceed max range in this tilt')
        self.drange = drange
        self.elev = self.el[tilt]
        try:
            data = np.ma.array(self.data[tilt][dtype])
        except KeyError:
            raise RadarDecodeError('Invalid product name')
        cut = data.T[:int(np.round(drange / reso))]
        shape_diff = np.round(drange / reso) - cut.shape[0]
        append = np.zeros((int(np.round(shape_diff)), cut.shape[1])) * np.ma.masked
        if dtype in ['VEL', 'SW']:
            try:
                rf = self.data[tilt]['RF']
            except KeyError:
                pass
            else:
                rf_flag = True
                rf = rf.T[:int(np.round(drange / reso))]
                rf = np.ma.vstack([rf, append])
        #r = np.ma.array(cut, mask=np.isnan(cut))
        r = np.ma.vstack([cut, append])
        if rf_flag:
            r.mask = np.logical_or(r.mask, ~rf.mask)
            ret = (r.T, rf.T)
        else:
            ret = r.T
        r_obj = Radial(ret, int(np.round(r.shape[0] * reso)), self.elev, reso, self.code, self.name, self.scantime, dtype,
                       self.stationlon, self.stationlat, nyquist_velocity=self.nyquist_v[tilt])
        x, y, z, d, a = self.projection(reso)
        r_obj.add_geoc(x, y, z)
        r_obj.add_polarc(d, a)
        if self.radartype == 'CC':
            r_obj.a_reso = 512
        return r_obj

    def projection(self, reso:float, h_offset:bool=False) -> tuple:
        r'''Calculate the geographic coordinates of the requested data range.'''
        theta = self.get_azimuth_angles(self.tilt)
        r = self.get_range(self.drange, reso)
        lonx, latx = get_coordinate(r, theta, self.elev, self.stationlon, self.stationlat, h_offset=h_offset)
        hght = height(r, self.elev, self.radarheight) * np.ones(theta.shape[0])[:, np.newaxis]
        return lonx, latx, hght, r, theta

    def to_nc(self, file:str, tilt:Union[str, int]='all', distance:number_type=230):
        r'''Store data in NetCDF format'''
        ds = NetCDFWriter(file)
        for i in range(self.get_nscans()):
            ds.create_radial(self.get_data(i, distance, 'REF'))
        ds._create_attribute('Time', self.scantime.strftime('%Y%m%d%H%M%S'))
        ds._create_attribute('Station Code', self.code)
        ds._create_attribute('Station Name', self.name)
        ds._create_attribute('Radar Type', self.radartype)

class StandardData(BaseRadar):
    r'''
    Class handling new cinrad standard data reading

    Attributes
    ----------
    scantime: datetime.datetime
        time of scan for this data
    code: str
        code for this radar
    angleindex_r: list
        indices of tilts which have reflectivity data
    angleindex_v: list
        indices of tilts which have velocity data
    stationlon: float
        logitude of this radar
    stationlat: float
        latitude of this radar
    radarheight: float
        height of this radar
    name: str
        name of this radar
    Rreso: float
        radial resolution of reflectivity data
    Vreso: float
        radial resolution of velocity data
    a_reso: int
        number of radials in one scan
    el: np.ndarray
        elevation angles for tilts
    drange: float
        current selected radius of data
    elev: float
        elevation angle of current selected tilt
    tilt: int
        current selected tilt
    '''
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 32:'Zc', 33:'Vc', 34:'Wc', 35:'ZDRc'}

    def __init__(self, file:Any):
        r'''
        Parameters
        ----------
        file: str
            path directed to the file to read
        '''
        self.f = prepare_file(file)
        self._parse()
        self.f.close()
        self.stationlat = self.geo['lat']
        self.stationlon = self.geo['lon']
        self.radarheight = self.geo['height']
        self.name = self.code
        self.angleindex_r = self.avaliable_tilt('REF') # API consistency
        del self.geo

    def _parse(self):
        header = np.frombuffer(self.f.read(32), SDD_header)
        if header['magic_number'] != 0x4D545352:
            raise RadarDecodeError('Invalid standard data')
        site_config = np.frombuffer(self.f.read(128), SDD_site)
        self.code = merge_bytes(site_config['site_code'][0])[:5].decode()
        self.geo = geo = dict()
        geo['lat'] = site_config['Latitude']
        geo['lon'] = site_config['Longitude']
        geo['height'] = site_config['ground_height']
        task = np.frombuffer(self.f.read(256), SDD_task)
        #task_name = merge_bytes(task['task_name'][0])
        self.scantime = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(task['scan_start_time']))
        cut_num = task['cut_number'][0]
        scan_config = np.frombuffer(self.f.read(256 * cut_num), SDD_cut)
        self.scan_config = [ScanConfig(*i) for i in scan_config]
        data = dict()
        aux = dict()
        if task['scan_type'] == 2: # Single-layer RHI
            self.scan_type = 'RHI'
        else:
            self.scan_type = 'PPI'
        while 1:
            radial_header = np.frombuffer(self.f.read(64), SDD_rad_header)
            el_num = radial_header['elevation_number'][0] - 1
            if el_num not in data.keys():
                data[el_num] = defaultdict(list)
                aux[el_num] = defaultdict(list)
            aux[el_num]['azimuth'].append(radial_header['azimuth'][0])
            aux[el_num]['elevation'].append(radial_header['elevation'][0])
            for _ in range(radial_header['moment_number'][0]):
                moment_header = np.frombuffer(self.f.read(32), SDD_mom_header)
                if moment_header[zip_type][0] == 1: # LZO compression
                    raise NotImplementedError('LZO compressed file is not supported')
                dtype = self.dtype_corr[moment_header['data_type'][0]]
                data_body = np.frombuffer(self.f.read(moment_header['block_length'][0]),
                                          'u{}'.format(moment_header['bin_length'][0]))
                if dtype not in aux[el_num].keys():
                    scale = moment_header['scale'][0]
                    offset = moment_header['offset'][0]
                    aux[el_num][dtype] = (scale, offset)
                data[el_num][dtype].append(data_body.tolist())
            if radial_header['radial_state'][0] in [4, 6]: # End scan
                break
        self.data = data
        self.aux = aux
        self.el = [i.elev for i in self.scan_config]

    def get_data(self, tilt:int, drange:number_type, dtype:str) -> Radial:
        r'''
        Get radar raw data

        Parameters
        ----------
        tilt: int
            index of elevation angle
        drange: float
            radius of data
        dtype: str
            type of product (REF, VEL, etc.)

        Returns
        -------
        r_obj: cinrad.datastruct.Radial
        '''
        self.tilt = tilt if self.scan_type == 'PPI' else 0
        self.drange = drange
        if self.scan_type == 'RHI':
            max_range = self.scan_config[0].max_range1 / 1000
            if drange > max_range:
                drange = max_range
        self.elev = self.el[tilt]
        try:
            raw = np.array(self.data[tilt][dtype])
        except KeyError:
            raise RadarDecodeError('Invalid product name')
        if raw.size == 0:
            warnings.warn('Empty data', RuntimeWarning)
        data = np.ma.array(raw, mask=(raw <= 5))
        reso = self.scan_config[tilt].dop_reso / 1000
        cut = data[:, :int(drange / reso)]
        shape_diff = np.round(drange / reso) - cut.shape[1]
        append = np.zeros((cut.shape[0], int(shape_diff))) * np.ma.masked
        if dtype in ['VEL', 'SW']:
            rf = np.ma.array(cut.data, mask=(cut.data != 1))
            rf = np.ma.hstack([rf, append])
        cut = np.ma.hstack([cut, append])
        scale, offset = self.aux[tilt][dtype]
        r = (cut - offset) / scale
        if dtype in ['VEL', 'SW']:
            ret = (r, rf)
        else:
            ret = r
        if self.scan_type == 'PPI':
            r_obj = Radial(ret, int(r.shape[1] * reso), self.elev, reso, self.code, self.name, self.scantime, dtype,
                           self.stationlon, self.stationlat, nyquist_velocity=self.scan_config[tilt].nyquist_spd)
            x, y, z, d, a = self.projection(reso)
            r_obj.add_geoc(x, y, z)
            r_obj.add_polarc(d, a)
            return r_obj
        else:
            # Manual projection
            dist = np.linspace(reso, self.drange, cut.shape[1])
            d, e = np.meshgrid(dist, self.aux[tilt]['elevation'])
            h = height(d, e, 0)
            rhi = Slice_(ret, d, h, self.scantime, self.code, self.name, dtype, azimuth=self.aux[tilt]['azimuth'][0])
            return rhi

    def projection(self, reso:float) -> tuple:
        r = np.arange(reso, self.drange + reso, reso)
        theta = np.array(self.aux[self.tilt]['azimuth']) * deg2rad
        lonx, latx = get_coordinate(r, theta, self.elev, self.stationlon, self.stationlat)
        hght = height(r, self.elev, self.radarheight) * np.ones(theta.shape[0])[:, np.newaxis]
        return lonx, latx, hght, r, theta

    def avaliable_tilt(self, product:str) -> List[int]:
        r'''Get all avaliable tilts for given product'''
        tilt = list()
        for i in list(self.data.keys()):
            if product in self.data[i].keys():
                tilt.append(i)
        return tilt

class NexradL2Data:
    r'''
    Class handling dual-polarized radar data (stored in Nexrad level II format) reading and plotting
    '''
    def __init__(self, file:Any):
        r'''
        Parameters
        ----------
        file: str
            path directed to the file to read
        '''
        from metpy.io.nexrad import Level2File
        self.f = Level2File(file)
        self.scantime = self.f.dt
        self.name = self.f.stid.decode()
        self.el = np.array([ray[0][0].el_angle for ray in self.f.sweeps])
        self.stationlon = self.f.sweeps[0][0][1].lon
        self.stationlat = self.f.sweeps[0][0][1].lat

    def get_data(self, tilt:int, drange:number_type, dtype:str) -> Radial:
        if isinstance(dtype, str):
            self.dtype = dtype.upper().encode()
        elif isinstance(dtype, bytes):
            self.dtype = dtype.upper()
        if self.dtype in [b'REF', b'VEL', b'ZDR', b'PHI', b'RHO']:
            if self.dtype in [b'ZDR', b'PHI', b'RHO'] and tilt in [1, 3]:
                tilt -= 1
                warnings.warn('Elevation angle {} does not contain {} data, automatically switch to tilt {}'.format(
                    tilt + 1, self.dtype.decode(), tilt))
            elif self.dtype in [b'VEL', b'SW'] and tilt in [0, 2]:
                tilt += 1
                warnings.warn('Elevation angle {} does not contain {} data, automatically switch to tilt {}'.format(
                    tilt - 1, self.dtype.decode(), tilt))
            hdr = self.f.sweeps[tilt][0][4][self.dtype][0]
            self.reso = hdr.gate_width
            raw = np.array([ray[4][self.dtype][1] for ray in self.f.sweeps[tilt]])
        else:
            raise RadarDecodeError('Unsupported data type {}'.format(self.dtype.decode()))
        cut = raw[:, :int(drange / self.reso)]
        masked = np.ma.array(cut, mask=np.isnan(cut))
        self.tilt = tilt
        self.drange = drange
        self.elev = self.el[tilt]
        x, y, z, d, a = self.projection(self.reso)
        radial = Radial(masked, drange, self.elev, self.reso, self.name, self.name, self.scantime, self.dtype.decode(),
                        self.stationlon, self.stationlat, x, y, a_reso=720)
        radial.add_polarc(d, a)
        return radial

    def projection(self, reso:float, h_offset:bool=False) -> tuple:
        header = self.f.sweeps[self.tilt][0][4][self.dtype][0]
        gatenum = header.num_gates
        firstgate = header.first_gate
        data_range = np.arange(gatenum) * reso + firstgate
        azi = np.array([ray[0].az_angle for ray in self.f.sweeps[self.tilt]]) * deg2rad
        datalength = int(self.drange / reso)
        lonx, latx = get_coordinate(data_range[:datalength], azi, self.elev, self.stationlon, self.stationlat,
                                    h_offset=h_offset)
        hght = height(data_range[:datalength], self.elev, 0) * np.ones(azi.shape[0])[:, np.newaxis]
        return lonx, latx, hght, data_range[:datalength], azi

class PUP(BaseRadar):
    r'''
    Class handling PUP data (Nexrad Level III data)
    '''
    def __init__(self, file:Any):
        from metpy.io.nexrad import Level3File
        f = Level3File(file)
        self.dtype = self._det_product_type(f.prod_desc.prod_code)
        self.radial_flag = self._is_radial()
        data_block = f.sym_block[0][0]
        data = np.ma.array(data_block['data']) # First element in data is mysteriously empty
        data[data == 0] = np.ma.masked
        self.data = np.ma.masked_invalid(f.map_data(data))
        self.max_range = f.max_range
        if self.radial_flag:
            self.az = np.array(data_block['start_az']) * deg2rad
            self.rng = np.linspace(1, f.max_range, data.shape[-1])
        else:
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

    def get_data(self) -> Grid:
        if self.radial_flag:
            lon, lat = self.projection()
            return Radial(self.data, self.max_range, self.el, 1, self.code, self.name, self.scantime,
                          self.dtype, self.stationlon, self.stationlat, lon, lat)
        else:
            return Grid(self.data, self.max_range, self.reso, self.code, self.name, self.scantime,
                        self.dtype, self.lon, self.lat)

    def _is_radial(self) -> bool:
        return self.dtype in range(16, 22) or self.dtype in range(22, 28) or self.dtype in range(28, 31) 

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
        header = np.frombuffer(f.read(1024), swan_header_dtype)
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
        self.data = np.ma.array((out - 66) / 2, mask=(out==0))

    def get_data(self) -> Grid:
        x, y = np.meshgrid(self.lon, self.lat)
        grid = Grid(self.data, np.nan, np.nan, 'SWAN', 'SWAN', self.data_time, self.product_name, x, y)
        return grid