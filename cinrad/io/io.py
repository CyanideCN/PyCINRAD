# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
import warnings
import datetime
from pathlib import Path
import bz2

import numpy as np

from cinrad.constants import deg2rad, con, con2, rm, MODULE_DIR
from cinrad.datastruct import Radial, _Slice
from cinrad.projection import get_coordinate, height
from cinrad.error import RadarDecodeError
#from cinrad.utils import _find_azimuth_position
from cinrad.io._io import NetCDFWriter
from cinrad.io.base import BaseRadar, _get_radar_info
from cinrad.io._dtype import SAB_dtype, CAB_dtype, CC_param, CC_data, CC_header

__all__ = ['CinradReader', 'StandardData', 'NexradL2Data', 'PUP']

def _detect_radartype(f, filename, type_assert=None):
    r'''Detect radar type from records in file'''
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
    def __init__(self, file, radar_type=None):
        r'''
        Parameters
        ----------
        file: str / obj with read method
            path directed to the file to read / file object
        radar_type: str, optional
            type of radar
        '''
        if hasattr(file, 'read'):
            f = file
            self.code, radartype = _detect_radartype(f, '')
        else:
            path = Path(file)
            filename = path.name
            filetype = path.suffix
            if filetype.lower().endswith('bz2'):
                f = bz2.open(file, 'rb')
            else:
                f = open(file, 'rb')
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

    def _SAB_handler(self, f, SAB=True):
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
        self.nyquist_v = data['nyquist_vel'][0] / 100
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
            out_data[idx]['REF'] = (np.ma.array(da['ref'], mask=(da['ref'] == 0)) - 2) / 2 - 32
            v = np.ma.array(da['vel'], mask=(da['vel'] < 2))
            if dv == 2:
                out_data[idx]['VEL'] = (v - 2) / 2 - 63.5
            elif dv == 4:
                out_data[idx]['VEL'] = v - 2 - 127
            out_data[idx]['SW'] = da['sw']
            out_data[idx]['azimuth'] = self.azimuth[b[idx]:b[idx + 1]]
            out_data[idx]['RF'] = np.ma.array(da['vel'], mask=(da['vel'] != 1))
        angleindex = np.arange(0, data['el_num'][-1], 1)
        self.angleindex_r = np.delete(angleindex, [1, 3])
        self.angleindex_v = np.delete(angleindex, [0, 2])
        self.data = out_data

    def _CC_handler(self, f):
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

    def _SC_handler(self, f):
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
        self.angleindex_r = self.angleindex_v = [i for i in range(len(self.el))]

    def get_nrays(self, scan):
        r'''Get number of radials in certain scan'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            return len(self.data[scan]['azimuth'])
        elif self.radartype == 'CC':
            return 512
        elif self.radartype == 'SC':
            return 360

    def get_azimuth_angles(self, scans=None):
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

    def get_elevation_angles(self, scans=None):
        if scans is None:
            return self.elevdeg
        else:
            return self.elevdeg[scans]

    def get_data(self, tilt, drange, dtype):
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
        if dtype == 'VEL':
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
                       self.stationlon, self.stationlat)
        x, y, z, d, a = self.projection(reso)
        r_obj.add_geoc(x, y, z)
        r_obj.add_polarc(d, a)
        if self.radartype == 'CC':
            r_obj.a_reso = 512
        return r_obj

    def projection(self, reso, h_offset=False):
        r'''Calculate the geographic coordinates of the requested data range.'''
        theta = self.get_azimuth_angles(self.tilt)
        r = self.get_range(self.drange, reso)
        lonx, latx = get_coordinate(r, theta, self.elev, self.stationlon, self.stationlat, h_offset=h_offset)
        hght = height(r, self.elev, self.radarheight) * np.ones(theta.shape[0])[:, np.newaxis]
        return lonx, latx, hght, r, theta

    def to_nc(self, file, tilt='all', distance=230):
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

    def __init__(self, file):
        r'''
        Parameters
        ----------
        file: str
            path directed to the file to read
        '''
        if hasattr(file, 'read'):
            f = file
        else:
            if file.lower().endswith('bz2'):
                f = bz2.open(file, 'rb')
            else:
                f = open(file, 'rb')
        f.seek(32)
        self.code = f.read(5).decode()
        f.seek(332)
        seconds = np.frombuffer(f.read(4), 'u4')[0]
        self.scantime = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(seconds))
        self.scanconfig = self._parse_configuration(f)
        self.data = self._parse_datablock(f)
        self.el = [self.scanconfig[i]['elevation_angle'] for i in self.scanconfig.keys()]
        self._update_radar_info()
        self.angleindex_r = self.avaliable_tilt('REF')
        f.close()

    @staticmethod
    def _parse_configuration(f):
        f.seek(336)
        scan_num = np.frombuffer(f.read(4), 'u4')[0]
        f.seek(416)
        config = dict()
        for i in range(scan_num):
            f.seek(24, 1)
            config[i] = dict()
            config[i]['elevation_angle'] = np.frombuffer(f.read(4), 'f4')[0]
            f.seek(8, 1)
            config[i]['angular_reso'] = np.frombuffer(f.read(4), 'f4')[0]
            f.seek(8, 1)
            config[i]['radial_reso'] = np.frombuffer(f.read(4), 'u4')[0] / 1000
            config[i]['max_distance'] = np.frombuffer(f.read(4), 'u4')[0] / 1000
            f.seek(200, 1)
        return config

    def _parse_datablock(self, f):
        data = dict()
        while True:
            header = f.read(64)#径向头块
            radial_state = np.frombuffer(header[0:4], 'u4')[0]
            el_num = np.frombuffer(header[16:20], 'u4')[0] - 1#仰角序号
            if el_num not in data.keys():
                data[el_num] = dict()
                data[el_num]['azimuth'] = list()
            #el = np.frombuffer(header[24:28], 'f4')[0]#仰角值
            az_num = np.frombuffer(header[20:24], 'f4')[0]#方位角
            #length = np.frombuffer(header[36:40], 'u4')[0]#数据块长度
            type_num = np.frombuffer(header[40:44], 'u4')[0]#数据类别数量
            for i in range(type_num):
                data_header = f.read(32)#径向数据头
                data_type = np.frombuffer(data_header[0:4], 'u4')[0]#数据类型
                scale = np.frombuffer(data_header[4:8], 'u4')[0]
                offset = np.frombuffer(data_header[8:12], 'u4')[0]
                bitlength = np.frombuffer(data_header[12:14], 'u2')[0]
                blocklength = np.frombuffer(data_header[16:20], 'u4')[0]
                body = f.read(blocklength)#径向数据
                raw = np.frombuffer(body, 'u' + str(bitlength)).astype(float)
                raw[raw == 0] = np.nan
                value = (raw - offset) / scale
                if self.dtype_corr[data_type] not in data[el_num].keys():
                    data[el_num][self.dtype_corr[data_type]] = list()
                data[el_num][self.dtype_corr[data_type]].append(value)
            data[el_num]['azimuth'].append(az_num)
            if radial_state == 4:
                break
        return data

    def get_data(self, tilt, drange, dtype):
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
        self.tilt = tilt
        self.drange = drange
        self.elev = self.el[tilt]
        try:
            data = np.array(self.data[tilt][dtype])
        except KeyError:
            raise RadarDecodeError('Invalid product name')
        if data.size == 0:
            raise RadarDecodeError('Current elevation does not contain this data.')
        reso = self.scanconfig[tilt]['radial_reso']
        cut = data[:, :int(drange / reso)]
        r = np.ma.array(cut, mask=np.isnan(cut))
        r_obj = Radial(r, int(r.shape[0] * reso), self.elev, reso, self.code, self.name, self.scantime, dtype,
                       self.stationlon, self.stationlat)
        x, y, z, d, a = self.projection(reso)
        r_obj.add_geoc(x, y, z)
        r_obj.add_polarc(d, a)
        return r_obj

    def projection(self, reso):
        r = np.arange(reso, self.drange + reso, reso)
        theta = np.array(self.data[self.tilt]['azimuth']) * deg2rad
        lonx, latx = get_coordinate(r, theta, self.elev, self.stationlon, self.stationlat)
        hght = height(r, self.elev, self.radarheight) * np.ones(theta.shape[0])[:, np.newaxis]
        return lonx, latx, hght, r, theta

    def avaliable_product(self, tilt):
        r'''Get all avaliable products in given tilt'''
        return list(self.data[tilt].keys())

    def avaliable_tilt(self, product):
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
    def __init__(self, file):
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

    def get_data(self, tilt, drange, dtype):
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
            raise RadarError('Unsupported data type {}'.format(self.dtype.decode()))
        cut = raw[:, :int(drange / self.reso)]
        masked = np.ma.array(cut, mask=np.isnan(cut))
        self.tilt = tilt
        self.drange = drange
        self.elev = self.el[tilt]
        x, y, z, d, a = self.projection(self.reso)
        radial = Radial(masked, drange, self.elev, self.reso, self.name, self.name, self.scantime, self.dtype.decode(),
                        self.stationlon, self.stationlat, x, y, a_reso=720)
        return radial

    def projection(self, reso, h_offset=False):
        header = self.f.sweeps[self.tilt][0][4][self.dtype][0]
        gatenum = header.num_gates
        firstgate = header.first_gate
        data_range = np.arange(gatenum) * reso + firstgate
        azi = np.array([ray[0].az_angle for ray in self.f.sweeps[self.tilt]]) * deg2rad
        datalength = int(self.drange / reso)
        lonx, latx = get_coordinate(data_range[:datalength], azi, self.elev, self.stationlon, self.stationlat,
                                    h_offset=h_offset)
        hght = height(data_range[:datalength], self.elev, 0) * np.ones(azi.shape[0])[:, np.newaxis]
        return lonx, latx, hght, data_range, azi

class PUP:
    r'''
    Class handling PUP data (Nexrad Level III data)
    Currently only radial data are supported
    '''
    def __init__(self, file):
        from metpy.io.nexrad import Level3File
        f = Level3File(file)
        data_block = f.sym_block[0][0]
        data = np.ma.array(data_block['data'][1:]) # First element in data is mysteriously empty
        data[data == 0] = np.ma.masked
        self.az = np.array(data_block['start_az'][:-1]) * deg2rad
        self.rng = np.linspace(1, f.max_range, data.shape[-1])
        self.data = f.map_data(data)
        self.stationlat = f.lat
        self.stationlon = f.lon
        self.el = f.metadata['el_angle']
        self.scantime = f.metadata['vol_time']
        o = open(file, 'rb')
        spec = np.frombuffer(o.read(2), '>i2')[0]
        self.dtype = self._det_product_type(spec)
        o.seek(12)
        self.code = 'Z9{}'.format(np.frombuffer(o.read(2), '>i2')[0])
        o.close()
        self._update_radar_info()

    def set_radarheight(self, height):
        self.radarheight = height

    def set_station_position(self, stationlon, stationlat):
        self.stationlon = stationlon
        self.stationlat = stationlat

    def set_station_name(self, name):
        self.name = name

    def _update_radar_info(self):
        r'''Update radar station info automatically.'''
        info = _get_radar_info(self.code)
        if info is None:
            warnings.warn('Auto fill radar station info failed, please set code manually', UserWarning)
        else:
            self.set_station_position(info[1], info[2])
            self.set_station_name(info[0])
            self.set_radarheight(info[4])

    def get_data(self):
        lon, lat = get_coordinate(self.rng, self.az, self.el, self.stationlon, self.stationlat, h_offset=False)
        return Radial(self.data, int(self.rng[-1]), self.el, 1, self.code, self.name, self.scantime.strftime('%Y%m%d%H%M%S'),
                      self.dtype, self.stationlon, self.stationlat, lon, lat)

    @staticmethod
    def _det_product_type(spec):
        if spec in range(16, 22):
            return 'REF'
        elif spec in range(22, 28):
            return 'VEL'
        elif spec in range(28, 31):
            return 'SW'
        else:
            raise RadarDecodeError('Unsupported product type {}, currently only radial\
                                    data are supported'.format(spec))