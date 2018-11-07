# -*- coding: utf-8 -*-
# Author: Puyuan Du

from .constants import deg2rad, con, con2, Rm1, modpath
from .datastruct import Radial, _Slice
from .projection import get_coordinate, height
from .error import RadarDecodeError

import os
import warnings
import datetime
from pathlib import Path

import numpy as np

__all__ = ['CinradReader', 'StandardData', 'DualPolRadar']

radarinfo = np.load(os.path.join(modpath, 'RadarStation.npy'))

def _get_radar_info(code):
    r'''Get radar station info from the station database according to the station code.'''
    try:
        pos = np.where(radarinfo[0] == code)[0][0]
    except IndexError:
        raise RadarDecodeError('Invalid radar code')
    name = radarinfo[1][pos]
    lon = radarinfo[2][pos]
    lat = radarinfo[3][pos]
    radartype = radarinfo[4][pos]
    radarheight = radarinfo[5][pos]
    return name, lon, lat, radartype, radarheight

class CinradReader:
    r'''
    Class handling CINRAD radar reading
    
    Parameters
    ----------
    filepath: str
        path directed to the file to read
    radar_type: str, optional
        type of radar
    '''
    def __init__(self, filepath, radar_type=None):
        path = Path(filepath)
        filename = path.name
        filetype = path.suffix
        if filetype.endswith('bz2'):
            import bz2
            f = bz2.open(filepath, 'rb')
        else:
            f = open(filepath, 'rb')
        radartype = self._detect_radartype(f, filename, type_assert=radar_type)
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

    def __lt__(self, value):
        return self.scantime < value.scantime

    def _detect_radartype(self, f, filename, type_assert=None):
        r'''Detect radar type from records in file'''
        f.seek(100)
        typestring = f.read(9)
        det_sc = typestring == b'CINRAD/SC'
        det_cd = typestring == b'CINRAD/CD'
        f.seek(116)
        det_cc = f.read(9) == b'CINRAD/CC'
        #Read information from filenames (if applicable)
        if filename.startswith('RADA'):
            spart = filename.split('-')
            self.code = spart[1]
            radartype = spart[2]
        elif filename.startswith('Z'):
            spart = filename.split('_')
            self.code = spart[3]
            radartype = spart[7]
        else:
            self.code = None
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
        return radartype

    def _SAB_handler(self, f, SAB=True):
        vraw = list()
        rraw = list()
        if SAB:
            blocklength = 2432
        else:
            blocklength = 4132
        copy = f.read(blocklength)
        f.seek(0)
        datalength = len(f.read())
        num = int(datalength / blocklength)
        azimuthx = list()
        eleang = list()
        boundary = list()
        count = 0
        deltdays = np.frombuffer(copy[32:34], 'u2')[0]
        deltsecs = np.frombuffer(copy[28:32], 'u4')[0]
        start = datetime.datetime(1969, 12, 31)
        deltday = datetime.timedelta(days=int(deltdays))
        deltsec = datetime.timedelta(milliseconds=int(deltsecs))
        self.scantime = start + deltday + deltsec
        self.vcpmode = np.frombuffer(copy[72:74], 'u2')[0]
        self.Rreso = np.frombuffer(copy[50:52], 'u2')[0] / 1000
        self.Vreso = np.frombuffer(copy[52:54], 'u2')[0] / 1000
        f.seek(98)
        self.code = f.read(5).decode()
        f.seek(0)
        while count < num:
            a = f.read(blocklength)
            azimuth = np.frombuffer(a[36:38], 'u2')
            datacon = np.frombuffer(a[40:42], 'u2')
            elevangle = np.frombuffer(a[42:44], 'u2')
            anglenum = np.frombuffer(a[44:46], 'u2')
            veloreso = np.frombuffer(a[70:72], 'u2')
            if SAB:
                R = np.frombuffer(a[128:588], 'u1')
                V = np.frombuffer(a[128:1508], 'u1')
            else:
                R = np.frombuffer(a[128:928], 'u1')
                V = np.frombuffer(a[128:2528], 'u1')
            azimuthx.append(azimuth[0])
            eleang.append(elevangle[0])
            vraw.append(V.tolist())
            rraw.append(R.tolist())
            if datacon[0] == 3:
                boundary.append(0)
            elif datacon[0] == 0:
                boundary.append(count)
            elif datacon[0] == 4:
                boundary.append(num - 1)
            count = count + 1
        rraw = np.array(rraw)
        vraw = np.array(vraw)
        self.el = np.array(eleang)[boundary][:-1] * con
        self.azimuth = np.array(azimuthx) * con * deg2rad
        dv = veloreso[0]
        r = np.ma.array(rraw, mask=(rraw == 0))
        r1 = (r - 2) / 2 - 32
        rf = np.ma.array(vraw, mask=(vraw != 1))
        v2 = np.ma.array(vraw, mask=(vraw == 0))
        if dv == 2:
            v2 = (v2 - 2) / 2 - 63.5
        elif dv == 4:
            v2 = (v2 - 2) - 127
        angleindex = np.arange(0, anglenum[0], 1)
        self.angleindex_r = np.delete(angleindex, [1, 3])
        self.angleindex_v = np.delete(angleindex, [0, 2])
        self.timestr = self.scantime.strftime('%Y%m%d%H%M%S')
        data = dict()
        for i in range(len(boundary) - 1):
            data[i] = dict()
            data[i]['REF'] = r1[boundary[i]:boundary[i + 1]]
            data[i]['VEL'] = v2[boundary[i]:boundary[i + 1]]
            data[i]['RF'] = rf[boundary[i]:boundary[i + 1]]
            data[i]['azimuth'] = self.azimuth[boundary[i]:boundary[i + 1]]
        self.data = data

    def _CC_handler(self, f):
        vraw = list()
        rraw = list()
        blocklength = 3000
        f.seek(0)
        datalength = len(f.read())
        num = int(datalength / blocklength)
        f.seek(106)
        self.code = f.read(10).decode().split('\x00')[0]
        f.seek(184)
        self.scantime = datetime.datetime(year=np.frombuffer(f.read(1), 'u1')[0] * 100 + np.frombuffer(f.read(1), 'u1')[0],
                                          month=np.frombuffer(f.read(1), 'u1')[0], day=np.frombuffer(f.read(1), 'u1')[0],
                                          hour=np.frombuffer(f.read(1), 'u1')[0], minute=np.frombuffer(f.read(1), 'u1')[0],
                                          second=np.frombuffer(f.read(1), 'u1')[0])
        count = 0
        f.seek(1024)
        while count < num:
            a = f.read(blocklength)
            r = np.frombuffer(a[:1000], np.short).astype(float)
            v = np.frombuffer(a[1000:2000], np.short).astype(float)
            rraw.append(r)
            vraw.append(v)
            count += 1
        rraw = np.array(rraw)
        r = np.ma.array(rraw, mask=(rraw == -32768)) / 10
        vraw = np.array(vraw)
        v = np.ma.array(vraw, mask=(vraw == -32768)) / 10
        self.Rreso = 0.3
        self.Vreso = 0.3
        self.el = [0.5, 1.45, 2.4, 3.4, 4.3, 6, 9.9, 14.6, 19.5]
        data = dict()
        for i in range(len(self.el) - 1):
            data[i] = dict()
            data[i]['REF'] = rraw[i * 512: (i + 1) * 512] / 10
            data[i]['VEL'] = v[i * 512: (i + 1) * 512]
            data[i]['azimuth'] = self.get_azimuth_angles(i)
        self.data = data
        self.timestr = self.scantime.strftime('%Y%m%d%H%M%S')
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
        self.timestr = self.scantime.strftime('%Y%m%d%H%M%S')

    def set_code(self, code):
        self.code = code
        self._update_radar_info()

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

    def get_nscans(self):
        return len(self.el)

    def get_nrays(self, scan):
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            return len(self.data[scan]['azimuth'])
        elif self.radartype == 'CC':
            return 512
        elif self.radartype == 'SC':
            return 360

    def get_azimuth_angles(self, scans=None):
        r'''Radian'''
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

    def get_range(self, drange, reso):
        return np.arange(reso, drange + reso, reso)

    def _find_azimuth_position(self, azimuth):
        r'''Find the relative position of a certain azimuth angle in the data array.'''
        count = 0
        self.azim = self.get_azimuth_angles(self.tilt) * deg2rad
        if azimuth < 0.3:
            azimuth = 0.5 #TODO: Fix bug occured when azimuth is smaller than 0.3 degree
        azimuth_ = azimuth * deg2rad
        a_sorted = np.sort(self.azim)
        add = False
        while count < len(self.azim):
            if azimuth_ == a_sorted[count]:
                break
            elif (azimuth_ - a_sorted[count]) * (azimuth_ - a_sorted[count + 1]) < 0:
                if abs((azimuth_ - a_sorted[count])) >= abs(azimuth_ - a_sorted[count + 1]):
                    add = True
                break
            count += 1
        if add:
            count += 1
        pos = np.where(self.azim == a_sorted[count])[0][0]
        return pos

    def get_data(self, tilt, drange, dtype):
        rf_flag = False
        self.tilt = tilt
        self.drange = drange
        self.elev = self.el[tilt]
        try:
            data = np.ma.array(self.data[tilt][dtype])
        except KeyError:
            raise RadarDecodeError('Invalid product name')
        reso = self.Rreso if dtype == 'REF' else self.Vreso
        length = data.shape[1] * reso
        cut = data.T[:int(drange / reso)]
        if dtype == 'REF': # Mask invalid reflectivity data
            c = np.copy(cut)
            c[c == np.ma.masked] = 0
            radialavr = [np.average(i) for i in c]
            threshold = 4
            g = np.gradient(radialavr)
            try:
                num = np.where(g[50:] > threshold)[0][0] + 50
                rm = cut[:num]
                nanmatrix = np.zeros((int(drange / self.Rreso) - num, cut.shape[1])) * np.ma.masked
                cut = np.ma.concatenate((rm, nanmatrix))
            except IndexError:
                pass
        if dtype == 'VEL':
            try:
                rf = self.data[tilt]['RF']
            except KeyError:
                pass
            else:
                rf_flag = True
                rf = rf.T[:int(drange / reso)]
        #r = np.ma.array(cut, mask=np.isnan(cut))
        r = cut
        if rf_flag:
            ret = (r.T, rf.T)
        else:
            ret = r.T
        r_obj = Radial(ret, int(r.shape[0] * reso), self.elev, reso, self.code, self.name, self.timestr, dtype,
                       self.stationlon, self.stationlat)
        x, y, z, d, a = self.projection(reso)
        r_obj.add_geoc(x, y, z)
        r_obj.add_polarc(d, a)
        if self.radartype == 'CC':
            r_obj.a_reso = 512
        return r_obj

    def projection(self, reso, h_offset=False):
        r'''Calculate the geographic coordinates of the requested data range.'''
        length = self.get_nrays(self.tilt)
        theta = self.get_azimuth_angles(self.tilt)
        r = self.get_range(self.drange, reso)
        lonx, latx = get_coordinate(r, theta, self.elev, self.stationlon, self.stationlat, h_offset=h_offset)
        hght = height(r, self.elev, self.radarheight) * np.ones(theta.shape[0])[:, np.newaxis]
        return lonx, latx, hght, r, theta

    def rhi(self, azimuth, drange, startangle=0, stopangle=9):
        r'''Clip the reflectivity data from certain elevation angles in a single azimuth angle.'''
        rhi = list()
        xcoor = list()
        ycoor = list()
        dist = np.arange(1, drange + 1, 1)
        for i in self.angleindex_r[startangle:stopangle]:
            cac = self.get_data(i, drange, 'REF').data
            pos = self._find_azimuth_position(azimuth)
            if pos is None:
                nanarray = np.zeros((drange))
                rhi.append(nanarray.tolist())
            else:
                rhi.append(cac[pos])
            theta = self.elev * deg2rad
            xcoor.append((dist * np.cos(theta)).tolist())
            ycoor.append(dist * np.sin(theta) + (dist ** 2 / (2 * Rm1 ** 2)).tolist())
        rhi = np.array(rhi)
        rhi[rhi < 0] = 0
        xc = np.array(xcoor)
        yc = np.array(ycoor)
        return _Slice(rhi, xc, yc, self.timestr, self.code, self.name, 'rhi', azimuth=azimuth,
                      drange=drange)

class StandardData:
    r'''
    Class handling new cinrad standard data reading
    
    Parameters
    ----------
    filepath: str
        path directed to the file to read
    '''
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 32:'Zc', 33:'Vc', 34:'Wc', 35:'ZDRc'}

    def __init__(self, filepath):
        if filepath.endswith('bz2'):
            import bz2
            f = bz2.open(filepath, 'rb')
        else:
            f = open(filepath, 'rb')
        f.seek(32)
        self.code = f.read(5).decode()
        f.seek(332)
        seconds = np.frombuffer(f.read(4), 'u4')[0]
        self.scantime = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(seconds))
        self.scanconfig = self._parse_configuration(f)
        self.timestr = self.scantime.strftime('%Y%m%d%H%M%S')
        self.data = self._parse_datablock(f)
        self.el = [0.50, 0.50, 1.45, 1.45, 2.4, 3.35, 4.3, 5.25, 6.2, 7.5, 8.7, 10, 12, 14, 16.7, 19.5]
        # TODO: Auto detect el angles
        self._update_radar_info()
        self.angleindex_r = self.avaliable_tilt('REF')

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

    def _find_azimuth_position(self, azimuth):
        r'''Find the relative position of a certain azimuth angle in the data array.'''
        count = 0
        azim = np.array(self.ad[self.level]) * deg2rad
        if azimuth < 0.3:
            azimuth = 0.5
        azimuth_ = azimuth * deg2rad
        a_sorted = np.sort(azim)
        add = False
        while count < len(azim):
            if azimuth_ == a_sorted[count]:
                break
            elif (azimuth_ - a_sorted[count]) * (azimuth_ - a_sorted[count + 1]) < 0:
                if abs((azimuth_ - a_sorted[count])) >= abs(azimuth_ - a_sorted[count + 1]):
                    add = True
                break
            count += 1
        if add:
            count += 1
        pos = np.where(azim == a_sorted[count])[0][0]
        return pos

    def _parse_configuration(self, f):
        f.seek(336)
        scan_num = np.frombuffer(f.read(4), 'u4')[0]
        f.seek(416)
        config = dict()
        for i in range(scan_num):
            f.seek(36, 1)
            config[i] = dict()
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
            el = np.frombuffer(header[24:28], 'f4')[0]#仰角值
            az_num = np.frombuffer(header[20:24], 'f4')[0]#方位角
            length = np.frombuffer(header[36:40], 'u4')[0]#数据块长度
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
        length = data.shape[1] * reso
        cut = data.T[:int(drange / reso)]
        r = np.ma.array(cut, mask=np.isnan(cut))
        r_obj = Radial(r.T, int(r.shape[0] * reso), self.elev, reso, self.code, self.name, self.timestr, dtype,
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

    def rhi(self, azimuth, drange, startangle=0, stopangle=10):
        r'''Clip the reflectivity data from certain elevation angles in a single azimuth angle.'''
        rhi = list()
        xcoor = list()
        ycoor = list()
        dist = np.arange(1, drange + 1, 1)
        for i in self.angleindex_r[startangle:stopangle]:
            cac = self.reflectivity(i, drange).data
            pos = self._find_azimuth_position(azimuth)
            if pos is None:
                nanarray = np.zeros((drange))
                rhi.append(nanarray.tolist())
            else:
                rhi.append(cac[pos])
            theta = self.elev * deg2rad
            xcoor.append((dist * np.cos(theta)).tolist())
            ycoor.append(dist * np.sin(theta) + (dist ** 2 / (2 * Rm1 ** 2)).tolist())
        rhi = np.array(rhi)
        xc = np.array(xcoor)
        yc = np.array(ycoor)
        return _Slice(rhi, xc, yc, self.timestr, self.code, self.name, 'rhi', azimuth=azimuth,
                      drange=drange)

    def avaliable_product(self, tilt):
        return list(self.data[tilt].keys())

    def avaliable_tilt(self, product):
        tilt = list()
        for i in list(self.data.keys()):
            if product in self.data[i].keys():
                tilt.append(i)
        return tilt

class DualPolRadar:
    r'''
    Class handling dual-polarized radar reading and plotting
    
    Parameters
    ----------
    filepath: str
        path directed to the file to read
    '''
    def __init__(self, filepath):
        from metpy.io.nexrad import Level2File
        self.f = Level2File(filepath)
        self.dtime = self.f.dt
        self.timestr = self.dtime.strftime('%Y%m%d%H%M%S')
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
        cut = raw.T[:int(drange / self.reso)]
        masked = np.ma.array(cut, mask=np.isnan(cut))
        self.tilt = tilt
        self.drange = drange
        self.elev = self.el[tilt]
        x, y, z, d, a = self.projection(self.reso)
        radial = Radial(masked.T, drange, self.elev, self.reso, self.name, self.name, self.timestr, self.dtype.decode(), 
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