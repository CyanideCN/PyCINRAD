# -*- coding: utf-8 -*-
#Author: Du puyuan

from .constants import deg2rad, con, con2, Rm1, modpath
from .datastruct import R, V, Section
from .projection import get_coordinate, height
from .error import RadarDecodeError

import os
import warnings
import datetime
from pathlib import Path

import numpy as np

radarinfo = np.load(os.path.join(modpath, 'RadarStation.npy'))

class CinradReader:
    r'''Class handling CINRAD radar reading'''
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
        if radar_type:
            if radartype is not radar_type:
                warnings.warn('Contradictory information from input radar type and\
                radar type detected from input file.')
            self.radartype = radar_type
        else:
            self.radartype = radartype

    def _detect_radartype(self, f, filename, type_assert=None):
        r'''Detect radar type from records in file'''
        f.seek(100)
        typestring = f.read(9)
        det_sc = typestring == b'CINRAD/SC'
        det_cd = typestring == b'CINRAD/CD'
        f.seek(116)
        det_cc = f.read(9) == b'CINRAD/CC'
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
        self.boundary = list()
        count = 0
        deltdays = np.fromstring(copy[32:34], 'u2')[0]
        deltsecs = np.fromstring(copy[28:32], 'u4')[0]
        start = datetime.datetime(1969, 12, 31)
        deltday = datetime.timedelta(days=int(deltdays))
        deltsec = datetime.timedelta(milliseconds=int(deltsecs))
        scantime = start + deltday + deltsec
        self.Rreso = 1
        self.Vreso = 0.25
        f.seek(98)
        self.code = f.read(5).decode()
        f.seek(0)
        while count < num:
            a = f.read(blocklength)
            azimuth = np.fromstring(a[36:38], 'u2')
            datacon = np.fromstring(a[40:42], 'u2')
            elevangle = np.fromstring(a[42:44], 'u2')
            anglenum = np.fromstring(a[44:46], 'u2')
            veloreso = np.fromstring(a[70:72], 'u2')
            if SAB:
                R = np.fromstring(a[128:588], 'u1')
                V = np.fromstring(a[128:1508], 'u1')
            else:
                R = np.fromstring(a[128:928], 'u1')
                V = np.fromstring(a[128:2528], 'u1')
            azimuthx.append(azimuth[0])
            eleang.append(elevangle[0])
            vraw.append(V.tolist())
            rraw.append(R.tolist())
            if datacon[0] == 3:
                self.boundary.append(0)
            elif datacon[0] == 0:
                self.boundary.append(count)
            elif datacon[0] == 4:
                self.boundary.append(num - 1)
            count = count + 1
        self.rraw = np.array(rraw)
        self.vraw = np.array(vraw)
        self.elevdeg = np.array(eleang) * con
        self.azimuth = np.array(azimuthx) * con * deg2rad
        self.dv = veloreso[0]
        angleindex = np.arange(0, anglenum[0], 1)
        self.angleindex_r = np.delete(angleindex, [1, 3])
        self.angleindex_v = np.delete(angleindex, [0, 2])
        self.elevindex = self.elevdeg[self.boundary][:-1]
        self.timestr = scantime.strftime('%Y%m%d%H%M%S')

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
        scantime = datetime.datetime(year=np.fromstring(f.read(1), 'u1')[0] * 100 + np.fromstring(f.read(1), 'u1')[0],
                                        month=np.fromstring(f.read(1), 'u1')[0], day=np.fromstring(f.read(1), 'u1')[0],
                                        hour=np.fromstring(f.read(1), 'u1')[0], minute=np.fromstring(f.read(1), 'u1')[0],
                                        second=np.fromstring(f.read(1), 'u1')[0])
        count = 0
        f.seek(1024)
        while count < num:
            a = f.read(blocklength)
            r = np.fromstring(a[:1000], np.short).astype(float)
            v = np.fromstring(a[1000:2000], np.short).astype(float)
            rraw.append(r)
            vraw.append(v)
            count += 1
        self.rraw = np.array(rraw)
        self.vraw = np.array(vraw)
        self.Rreso = 0.3
        self.Vreso = 0.3
        self.timestr = scantime.strftime('%Y%m%d%H%M%S')
        self.elevdeg = [0.5, 1.5, 2.4, 3.4, 4.3, 6, 9.89, 14.6, 19.5]

    def _SC_handler(self, f):
        vraw = list()
        rraw = list()
        blocklength = 4000
        utc_offset = datetime.timedelta(hours=8)
        f.seek(853)
        scantime = datetime.datetime(year=np.fromstring(f.read(2), 'u2')[0], month=np.fromstring(f.read(1), 'u1')[0],
                                        day=np.fromstring(f.read(1), 'u1')[0], hour=np.fromstring(f.read(1), 'u1')[0],
                                        minute=np.fromstring(f.read(1), 'u1')[0], second=np.fromstring(f.read(1), 'u1')[0]) - utc_offset
        f.seek(1024)
        self.Rreso = 0.3
        self.Vreso = 0.3
        elev = list()
        count = 0
        while count < 3240:
            q = f.read(blocklength)
            elev.append(np.fromstring(q[2:4], 'u2')[0])
            x = np.fromstring(q[8:], 'u1').astype(float)
            rraw.append(x[slice(None, None, 4)])
            vraw.append(x[slice(1, None, 4)])
            count += 1
        self.rraw = np.concatenate(rraw).reshape(3240, 998)
        self.vraw = np.concatenate(vraw).reshape(3240, 998)
        self.elevdeg = np.array(elev[slice(359, None, 360)]) * con2
        self.azimuth = np.arange(0, 360, 1) * deg2rad
        self.timestr = scantime.strftime('%Y%m%d%H%M%S')

    def set_station_position(self, stationlon, stationlat):
        self.stationlon = stationlon
        self.stationlat = stationlat

    def set_station_name(self, name):
        self.name = name

    def set_drange(self, drange):
        self.drange = drange

    def set_code(self, code):
        self.code = code
        self._update_radar_info()

    def set_radarheight(self, height):
        self.radarheight = height

    def set_elevation_angle(self, angle):
        self.elev = angle

    def set_level(self, level):
        self.level = level

    def _get_radar_info(self):
        r'''Get radar station info from the station database according to the station code.'''
        if self.code is None:
            warnings.warn('Radar code undefined', UserWarning)
            return None
        try:
            pos = np.where(radarinfo[0] == self.code)[0][0]
        except IndexError:
            raise RadarDecodeError('Invalid radar code')
        name = radarinfo[1][pos]
        lon = radarinfo[2][pos]
        lat = radarinfo[3][pos]
        radartype = radarinfo[4][pos]
        radarheight = radarinfo[5][pos]
        return name, lon, lat, radartype, radarheight

    def _update_radar_info(self):
        r'''Update radar station info automatically.'''
        info = self._get_radar_info()
        if info is None:
            warnings.warn('Auto fill radar station info failed, please set code manually', UserWarning)
        else:
            self.set_station_position(info[1], info[2])
            self.set_station_name(info[0])
            self.set_radarheight(info[4])

    def _find_azimuth_position(self, azimuth):
        r'''Find the relative position of a certain azimuth angle in the data array.'''
        count = 0
        self.azim = self.azimuth[self.boundary[self.level]:self.boundary[self.level + 1]] * deg2rad
        if azimuth < 0.3:
            azimuth = 0.5
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
                else:
                    break
            count += 1
        if add:
            count += 1
        pos = np.where(self.azim == a_sorted[count])[0][0]
        return pos

    def reflectivity(self, level, drange):
        r'''Clip desired range of reflectivity data.'''
        self.level = level
        self.drange = drange
        length = self.rraw.shape[1] * self.Rreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.', UserWarning)
            self.drange = int(self.rraw.shape[1] * self.Rreso)
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            self.elev = self.elevdeg[self.boundary[level]]
            if level not in self.angleindex_r:
                warnings.warn('Use this elevation angle may yield unexpected result.', UserWarning)
            r = self.rraw[self.boundary[level]:self.boundary[level + 1]]
            r = (r - 2) / 2 - 32
            r1 = r.T[:int(drange / self.Rreso)]
        elif self.radartype == 'CC':
            self.elev = self.elevdeg[level]
            r1 = self.rraw[level * 512:(level + 1) * 512, :int(drange / self.Rreso)].T / 10
        elif self.radartype == 'SC':
            self.elev = self.elevdeg[level]
            r = self.rraw[level * 360:(level + 1) * 360, :int(drange / self.Rreso)].T
            r1 = (r - 64) / 2
        r1[r1 < 0] = 0
        radialavr = [np.average(i) for i in r1]
        threshold = 4
        g = np.gradient(radialavr)
        try:
            num = np.where(g[50:] > threshold)[0][0] + 50
            rm = r1[:num]
            nanmatrix = np.zeros((int(drange / self.Rreso) - num, r1.shape[1]))
            r1 = np.concatenate((rm, nanmatrix))
        except IndexError:
            pass
        r_obj = R(r1.T, drange, self.elev, self.Rreso, self.code, self.name, self.timestr,
                  self.stationlon, self.stationlat)
        x, y, z, d, a = self.projection('r')
        r_obj.add_geoc(x, y, z)
        r_obj.add_polarc(d, a)
        r_obj.a_reso = 512
        return r_obj

    def velocity(self, level, drange):
        r'''Clip desired range of velocity data.'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            if level not in self.angleindex_v:
                warnings.warn('Use this elevation angle may yield unexpected result.', UserWarning)
            self.elev = self.elevdeg[self.boundary[level]]
        self.drange = drange
        self.level = level
        length = self.vraw.shape[1] * self.Vreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.', UserWarning)
            self.drange = int(self.vraw.shape[1] * self.Vreso)
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            v = self.vraw[self.boundary[level]:self.boundary[level + 1]]
            v1 = v.T[:int(drange / self.Vreso)].astype(float)
            rf = np.ma.array(v1, mask=(v1 != 1))
            v1[v1 == 0] = np.nan
            if self.dv == 2:
                v1 = (v1 - 2) / 2 - 63.5
            elif self.dv == 4:
                v1 = (v1 - 2) - 127
            v_obj = V([v1.T, rf.T], drange, self.elev, self.Rreso, self.code, self.name, self.timestr,
                      self.stationlon, self.stationlat)
        elif self.radartype == 'CC':
            v = self.vraw[level * 512:(level + 1) * 512, :int(drange / self.Vreso)].T
            v[v == -32768] = np.nan
            v1 = v / 10
            v_obj = V(v1.T, drange, self.elev, self.Rreso, self.code, self.name, self.timestr
                      ,self.stationlon, self.stationlat, include_rf=False)
        elif self.radartype == 'SC':
            self.elev = self.elevindex[level]
            v = self.vraw[level * 360:(level + 1) * 360, :int(drange / self.Vreso)].T
            v[v == -64] = np.nan
            v1 = (v - 128) / 2
            v_obj = V(v1.T, drange, self.elev, self.Rreso, self.code, self.name,
                      self.timestr, self.stationlon, self.stationlat, include_rf=False)
        x, y, z, d, a = self.projection('v')
        v_obj.add_geoc(x, y, z)
        v_obj.add_polarc(d, a)
        return v_obj

    def projection(self, datatype, h_offset=False):
        r'''Calculate the geographic coordinates of the requested data range.'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            length = self.boundary[self.level + 1] - self.boundary[self.level]
        elif self.radartype == 'CC':
            length = 512
        elif self.radartype == 'SC':
            length = 360
        if datatype == 'r':
            r = np.arange(self.Rreso, self.drange + self.Rreso, self.Rreso)
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = self.azimuth[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype in ['CC', 'SC']:
                theta = np.linspace(0, 360, length) * deg2rad
        elif datatype == 'v':
            r = np.arange(self.Vreso, self.drange + self.Vreso, self.Vreso)
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = self.azimuth[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype in ['CC', 'SC']:
                theta = np.linspace(0, 360, length) * deg2rad
        elif datatype in ['et', 'vil']:
            r = np.arange(self.Rreso, self.drange + self.Rreso, self.Rreso)
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = np.arange(0, 361, 1) * deg2rad
            elif self.radartype in ['CC', 'SC']:
                theta = np.linspace(0, 360, length) * deg2rad
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
        rhi[rhi < 0] = 0
        xc = np.array(xcoor)
        yc = np.array(ycoor)
        return Section(rhi, xc, yc, azimuth, drange, self.timestr, self.code, self.name, 'rhi')