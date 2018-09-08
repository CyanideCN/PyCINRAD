# -*- coding: utf-8 -*-
#Author: Du puyuan

from form_colormap import form_colormap
import calc

import warnings
import datetime
from pathlib import Path
import json

import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cmx
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
from scipy.interpolate import griddata

mpl.rc('font', family='Arial')
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
con = (180 / 4096) * 0.125
con2 = 0.001824 # calculated manually
Rm1 = 8500
deg2rad = np.pi / 180
config = open('config.ini').read()
folderpath = json.loads(config)['filepath']

r_cmap = form_colormap('colormap\\r_main.txt', sep=True)
v_cmap = form_colormap('colormap\\v_main.txt', sep=False)
vel_cbar = form_colormap('colormap\\v_cbar.txt', sep=True)
rhi_cmap_smooth = form_colormap('colormap\\r_main.txt', sep=False, spacing='v')
r_cmap_smooth = form_colormap('colormap\\r_smooth.txt', sep=False, spacing='v')
zdr_cmap = form_colormap('colormap\\zdr_main.txt', sep=False)
zdr_cbar = form_colormap('colormap\\zdr_cbar.txt', sep=True)
kdp_cmap = form_colormap('colormap\\kdp_main.txt', sep=False)
kdp_cbar = form_colormap('colormap\\kdp_cbar.txt', sep=True)
cc_cmap = form_colormap('colormap\\cc_main.txt', sep=False)
cc_cbar = form_colormap('colormap\\cc_cbar.txt', sep=True)
et_cmap = form_colormap('colormap\\et.txt', sep=False)
et_cbar = form_colormap('colormap\\etbar.txt', sep=True)
radarinfo = np.load('RadarStation.npy')
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)

def check_radartype(accept_list):
    r'''Check if the decorated function is supported for this type of radar.'''
    def check(func):
        def inner(self, *args, **kwargs):
            if self.radartype not in accept_list:
                raise RadarError('{} radar is not supported for fuction {}'.format(self.radartype, func.__name__))
            return func(self, *args, **kwargs)
        return inner
    return check

class RadarError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class Radar:
    r'''Class handling CINRAD radar reading and plotting.'''
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
        if radartype in ['SA', 'SB']:
            self._SAB_handler(f)
        elif radartype in ['CA', 'CB']:
            self._SAB_handler(f, SAB=False)
        elif radartype == 'CC':
            self._CC_handler(f)
        elif radartype == 'SC':
            self._SC_handler(f)
        else:
            raise RadarError('Unrecognized data')
        self._update_radar_info()
        self.radartype = radartype

    def _detect_radartype(self, f, filename, type_assert=None):
        f.seek(100)
        typestring = f.read(9)
        det_sc = typestring == b'CINRAD/SC'
        det_cd = typestring == b'CINRAD/CD'
        f.seek(116)
        det_cc = f.read(9) == b'CINRAD/CC'
        f.seek(0)
        if filename.startswith('RADA'):
            spart = filename.split('-')
            self.code = spart[1]
            radartype = spart[2]
        elif filename.startswith('Z'):
            spart = filename.split('_')
            self.code = spart[3]
            radartype = spart[7]
        elif filetype.endswith('A'):
            self.code = None
            radartype = 'SA'
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
        deltdays = np.fromstring(copy[32:34], dtype='u2')[0]
        deltsecs = np.fromstring(copy[28:32], dtype='u4')[0]
        start = datetime.datetime(1969, 12, 31)
        deltday = datetime.timedelta(days=int(deltdays))
        deltsec = datetime.timedelta(milliseconds=int(deltsecs))
        scantime = start + deltday + deltsec
        self.Rreso = np.fromstring(copy[50:52], dtype='u2')[0] / 1000
        self.Vreso = np.fromstring(copy[52:54], dtype='u2')[0] / 1000
        f.seek(0)
        while count < num:
            a = f.read(blocklength)
            azimuth = np.fromstring(a[36:38], dtype='u2')
            datacon = np.fromstring(a[40:42], dtype='u2')
            elevangle = np.fromstring(a[42:44], dtype='u2')
            anglenum = np.fromstring(a[44:46], dtype='u2')
            veloreso = np.fromstring(a[70:72], dtype='u2')
            if SAB:
                R = np.fromstring(a[128:588], dtype='u1')
                V = np.fromstring(a[128:1508], dtype='u1')
            else:
                R = np.fromstring(a[128:928], dtype='u1')
                V = np.fromstring(a[128:2528], dtype='u1')
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
        self.z = np.array(eleang) * con
        self.aziangle = np.array(azimuthx) * con
        self.rad = self.aziangle * deg2rad
        self.vraw = np.array(vraw)
        self.dv = veloreso[0]
        anglelist = np.arange(0, anglenum[0], 1)
        self.anglelist_r = np.delete(anglelist, [1, 3])
        self.anglelist_v = np.delete(anglelist, [0, 2])
        self.elevanglelist = self.z[self.boundary][:-1]
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
        scantime = datetime.datetime(year=np.fromstring(f.read(1), dtype='u1')[0] * 100 + np.fromstring(f.read(1), dtype='u1')[0],
                                        month=np.fromstring(f.read(1), dtype='u1')[0], day=np.fromstring(f.read(1), dtype='u1')[0],
                                        hour=np.fromstring(f.read(1), dtype='u1')[0], minute=np.fromstring(f.read(1), dtype='u1')[0],
                                        second=np.fromstring(f.read(1), dtype='u1')[0])
        count = 0
        f.seek(1024)
        while count < num:
            a = f.read(blocklength)
            r = np.fromstring(a[:1000], dtype=np.short).astype(float)
            v = np.fromstring(a[1000:2000], dtype=np.short).astype(float)
            rraw.append(r)
            vraw.append(v)
            count += 1
        self.rraw = np.array(rraw)
        self.vraw = np.array(vraw)
        self.Rreso = 0.3
        self.Vreso = 0.3
        self.timestr = scantime.strftime('%Y%m%d%H%M%S')

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
        self.elevanglelist = np.array(elev[slice(359, None, 360)]) * con2
        self.aziangle = np.arange(0, 360, 1) * deg2rad
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

    def set_radar_height(self, height):
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
            raise RadarError('Invalid radar code')
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
            self.set_radar_height(info[4])

    def _height(self, distance, elevation):
        return distance * np.sin(elevation * deg2rad) + distance ** 2 / (2 * Rm1) + self.radarheight / 1000

    def _find_azimuth_position(self, azimuth):
        r'''Find the relative position of a certain azimuth angle in the data array.'''
        count = 0
        self.azim = self.aziangle[self.boundary[self.level]:self.boundary[self.level + 1]] * deg2rad
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
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            self.elev = self.z[self.boundary[level]]
            print(self.elev)
            if level in [1, 3]:
                warnings.warn('Use this elevation angle may yield unexpected result.', UserWarning)
        self.level = level
        self.drange = drange
        length = self.rraw.shape[1] * self.Rreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.', UserWarning)
            self.drange = int(self.rraw.shape[1] * self.Rreso)
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            dbz = (self.rraw - 2) / 2 - 32
            r = dbz[self.boundary[level]:self.boundary[level + 1]]
            r1 = r.transpose()[:int(drange / self.Rreso)]
        elif self.radartype == 'CC':
            dbz = self.rraw / 10
            r1 = dbz[level * 512:(level + 1) * 512, :int(drange / self.Rreso)].T
        elif self.radartype == 'SC':
            self.elev = self.elevanglelist[level]
            dbz = (self.rraw - 64) / 2
            r1 = dbz[level * 360:(level + 1) * 360, :int(drange / self.Rreso)].T
        r1[r1 < 0] = 0
        radialavr = list()
        for i in r1:
            radialavr.append(np.average(i))
        threshold = 4
        g = np.gradient(radialavr)
        try:
            num = np.where(g[50:] > threshold)[0][0] + 50
            rm = r1[:num]
            nanmatrix = np.zeros((int(drange / self.Rreso) - num, r1.shape[1]))
            r1 = np.concatenate((rm, nanmatrix))
        except IndexError:
            pass
        return r1.T

    def velocity(self, level, drange):
        r'''Clip desired range of velocity data.'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            if level in [0, 2]:
                warnings.warn('Use this elevation angle may yield unexpected result.', UserWarning)
            self.elev = self.z[self.boundary[level]]
            print(self.elev)
        self.drange = drange
        self.level = level
        length = self.vraw.shape[1] * self.Vreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.', UserWarning)
            self.drange = int(self.vraw.shape[1] * self.Vreso)
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            if self.dv == 2:
                v = (self.vraw - 2) / 2 - 63.5
            elif self.dv == 4:
                v = (self.vraw - 2) - 127
            v = v[self.boundary[level]:self.boundary[level + 1]]
            v1 = v.transpose()[:int(drange / self.Vreso)]
            v1[v1 == -64.5] = np.nan
            rf = np.ma.array(v1, mask=(v1 != -64))
            return v1.T, rf.T
        elif self.radartype == 'CC':
            v = self.vraw / 10
            v1 = v[level * 512:(level + 1) * 512, :int(drange / self.Vreso)].T
            v1[v1 == -3276.8] = np.nan
            return v1.T, None
        elif self.radartype == 'SC':
            self.elev = self.elevanglelist[level]
            v = (self.vraw - 128) / 2
            v1 = v[level * 360:(level + 1) * 360, :int(drange / self.Vreso)].T
            v1[v1 == -64] = np.nan
            return v1.T, None

    def _get_coordinate(self, distance, angle):
        r'''Convert polar coordinates to geographic coordinates with the given radar station position.'''
        if self.elev is None:
            raise RadarError('The elevation angle is not defined')
        elev = self.elev
        deltav = np.cos(angle[:, np.newaxis]) * distance * np.cos(np.deg2rad(elev))
        deltah = np.sin(angle[:, np.newaxis]) * distance * np.cos(np.deg2rad(elev))
        deltalat = deltav / 111
        actuallat = deltalat + self.stationlat
        deltalon = deltah / 111
        actuallon = deltalon + self.stationlon
        return actuallon, actuallat

    def projection(self, datatype):
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
                theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype == 'CC':
                theta = np.linspace(0, 360, length) * deg2rad
            elif self.radartype == 'SC':
                theta = np.linspace(0, 360, length) * deg2rad
        elif datatype == 'v':
            r = np.arange(self.Vreso, self.drange + self.Vreso, self.Vreso)
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype == 'CC':
                theta = np.linspace(0, 360, length) * deg2rad
            elif self.radartype == 'SC':
                theta = np.linspace(0, 360, length) * deg2rad
        elif datatype == 'et':
            r = np.arange(self.Rreso, self.drange + self.Rreso, self.Rreso)
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = np.arange(0, 361, 1) * deg2rad
            elif self.radartype == 'CC':
                theta = np.linspace(0, 360, 512) * deg2rad
            elif self.radartype == 'SC':
                theta = np.linspace(0, 360, 360) * deg2rad
        lonx, latx = self._get_coordinate(r, theta)
        height = self._height(r, self.elev) * np.ones(theta.shape[0])[:, np.newaxis]
        return lonx, latx, height

    def draw_ppi(self, level, drange, datatype, draw_author=True, smooth=False, dpi=350):
        r'''Plot reflectivity PPI scan with the default plot settings.'''
        suffix = ''
        calc_ = True
        if datatype == 'r':
            data = self.reflectivity(level, drange)
        elif datatype == 'v':
            data, rf = self.velocity(level, drange)
        elif datatype == 'et':
            data = self.echo_top(drange)
            self.set_elevation_angle(0)
        elif datatype == 'cr':
            lons, lats, data = self.composite_reflectivity(drange=drange)
            calc_ = False
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        if calc_:
            coor = self.projection(datatype)
            lons, lats = coor[0], coor[1]
        lonm, latm = np.max(lons), np.max(lats)
        x_delta = lonm - self.stationlon
        y_delta = latm - self.stationlat
        angle_offset = np.cos(self.elev * deg2rad)
        x_offset = x_delta / angle_offset
        y_offset = y_delta / angle_offset
        plt.style.use('dark_background')
        m = Basemap(llcrnrlon=self.stationlon - x_offset, urcrnrlon=self.stationlon + x_offset
                    , llcrnrlat=self.stationlat - y_offset, urcrnrlat=self.stationlat + y_offset, resolution="l")
        if datatype == 'r':
            typestring = 'Base Reflectivity'
            cmaps = r_cmap
            norms = norm1
            reso = self.Rreso
            r1 = data[np.logical_not(np.isnan(data))]
            if smooth:
                m.contourf(lons.flatten(), lats.flatten(), data.flatten(), 256, cmap=r_cmap_smooth, norm=norms, tri=True)
                suffix = '_smooth'
            else:
                data[data <= 2] = None
                m.pcolormesh(lons, lats, data, norm=norms, cmap=cmaps)
        elif datatype == 'v':
            typestring = 'Base Velocity'
            cmaps = vel_cbar
            norms = cmx.Normalize(0, 1)
            reso = self.Vreso
            m.pcolormesh(lons, lats, data, cmap=v_cmap, norm=norm2)
            rfmap = cmx.ListedColormap('#660066', '#FFFFFF')
            if rf is not None:
                m.pcolormesh(lons, lats, rf, cmap=rfmap, norm=cmx.Normalize(-1, 0))
        elif datatype == 'et':
            typestring = 'Echo Tops'
            cmaps = et_cbar
            norms = cmx.Normalize(0, 1)
            reso = self.Rreso
            data[data > 25] = 0
            m.pcolormesh(lons, lats, data, cmap=et_cmap, norm=cmx.Normalize(0, 21))
        elif datatype == 'cr':
            typestring = 'Composite Ref.'
            cmaps = r_cmap
            norms = norm1
            reso = 1
            self.set_elevation_angle(0)
            data[data <= 2] = None
            r1 = data[np.logical_not(np.isnan(data))]
            m.contourf(lons, lats, data, 128, norm=norms, cmap=cmaps)
        m.readshapefile('shapefile\\County', 'states', drawbounds=True, linewidth=0.5, color='grey')
        m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.7, color='lightgrey')
        m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=1, color='white')
        plt.axis('off')
        ax2 = fig.add_axes([0.92, 0.12, 0.04, 0.35])
        cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmaps, norm=norms, orientation='vertical', drawedges=False)
        cbar.ax.tick_params(labelsize=8)
        if datatype == 'v':
            cbar.set_ticks(np.linspace(0, 1, 16))
            cbar.set_ticklabels(['RF', '', '27', '20', '15', '10', '5', '1', '0', '-1', '-5', '-10', '-15', '-20', '-27', '-35'])
        elif datatype == 'et':
            cbar.set_ticks(np.linspace(0, 1, 16))
            cbar.set_ticklabels(['', '21', '20', '18', '17', '15', '14', '12', '11', '9', '8', '6', '5', '3', '2', '0'])
        ax2.text(0, 2.13, typestring, fontproperties=font2)
        ax2.text(0, 2.09, 'Range: {:.0f}km'.format(self.drange), fontproperties=font2)
        ax2.text(0, 2.05, 'Resolution: {:.2f}km'.format(reso) , fontproperties=font2)
        ax2.text(0, 2.01, 'Date: {}.{}.{}'.format(self.timestr[:4], self.timestr[4:6], self.timestr[6:8]), fontproperties=font2)
        ax2.text(0, 1.97, 'Time: {}:{}'.format(self.timestr[8:10], self.timestr[10:12]), fontproperties=font2)
        ax2.text(0, 1.93, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.89, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.85, 'Elev: {:.2f}deg'.format(self.elev), fontproperties=font2)
        if datatype in ['r', 'cr']:
            ax2.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(r1)), fontproperties=font2)
        elif datatype == 'et':
            ax2.text(0, 1.81, 'Max: {:.1f}km'.format(np.max(data)), fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.73, 'Made by HCl', fontproperties=font2)
        plt.savefig('{}{}_{}_{:.1f}_{}_{}{}.png'.format(
            folderpath, self.code, self.timestr, self.elev, self.drange, datatype.upper(), suffix), bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig

    def rhi(self, azimuth, drange, startangle=0, stopangle=9, height=15, interpolation=False):
        r'''Clip the reflectivity data from certain elevation angles in a single azimuth angle.'''
        rhi = list()
        xcoor = list()
        ycoor = list()
        dist = np.arange(1, drange + 1, 1)
        for i in self.anglelist_r[startangle:stopangle]:
            cac = self.reflectivity(i, drange)
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
        if interpolation:
            from metpy import gridding
            xi = np.arange(0, drange + 1, 1)
            yi = np.arange(0, height + 0.5, 0.5)
            x, y = np.meshgrid(xi, yi)
            z = gridding.natural_neighbor(xc.flatten(), yc.flatten(), rhi.flatten(), x, y)
            return x, y, z
        else:
            return xc, yc, rhi

    @check_radartype(['SA', 'SB', 'CB'])
    def draw_rhi(self, azimuth, drange, startangle=0, stopangle=8, height=15, interpolation=False):
        r'''Plot reflectivity RHI scan with the default plot settings.'''
        xc, yc, rhi = self.rhi(azimuth, drange, startangle=startangle, stopangle=stopangle
                               , height=height, interpolation=interpolation)
        rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 4), dpi=200)
        plt.contourf(xc, yc, rhi, 128, cmap=rhi_cmap_smooth, norm=norm1, corner_mask=False)
        plt.ylim(0, height)
        plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                  self.name, azimuth, self.timestr[:4], self.timestr[4:6], self.timestr[6:8], self.timestr[8:10]
                  , self.timestr[10:12], rmax), fontproperties=font2)
        plt.ylabel('Altitude (km)')
        plt.xlabel('Range (km)')
        plt.savefig('{}{}_{}_RHI_{}_{}.png'.format(folderpath, self.code, self.timestr, self.drange, azimuth)
                    , bbox_inches='tight')

    def _r_resample(self, drange=230):
        Rrange = np.arange(self.Rreso, drange + self.Rreso, self.Rreso)
        Trange = np.arange(0, 361, 1)
        dist, theta = np.meshgrid(Rrange, Trange)
        r_resampled = list()
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            anglelist = self.anglelist_r
        for i in anglelist:
            r = self.reflectivity(i, drange)
            azimuth = self.aziangle[self.boundary[i]:self.boundary[i + 1]]
            dist_, theta_ = np.meshgrid(Rrange, azimuth)
            r_ = griddata((dist_.flatten(), theta_.flatten()), r.flatten(), (dist, theta), method='nearest')
            r_resampled.append(r_)
        r_res = np.concatenate(r_resampled)
        return r_res.reshape(r_res.shape[0] // 361, 361, int(drange / self.Rreso)), dist, theta

    @check_radartype(['SA', 'SB'])
    def echo_top(self, drange, threshold=18.):
        data = self._r_resample(drange=drange)
        elev = np.delete(self.elevanglelist * deg2rad, [1, 3]).tolist()
        et = calc.echo_top(data, elev, self.radarheight, threshold)
        return et

    @check_radartype(['SA'])
    def _grid_3d(self, resolution=(100, 100, 20)):
        r'''Convert radar data to 3d grid (test)'''
        from xarray import DataArray
        r = self._r_resample()[0]
        phi = self.elevanglelist[self.anglelist_r]
        x = list()
        y = list()
        z = list()
        for i in phi:
            self.set_elevation_angle(i)
            x_, y_, z_ = self.projection(datatype='et')
            x.append(x_.reshape(361, 230))
            y.append(y_.reshape(361, 230))
            z.append(z_.reshape(361, 230))
        lon = np.array(x)
        lat = np.array(y)
        height = np.array(z)
        x_res, y_res, z_res = resolution
        grid_x, grid_y, grid_z = np.mgrid[np.min(x):np.max(x):x_res * 1j, np.min(y):np.max(y):y_res * 1j
                                            , 0:20:z_res * 1j]
        height_mask = np.ma.array(height, mask=(height<=20)).mask
        grid_r = griddata((lon[height_mask], lat[height_mask], height[height_mask]), r[height_mask], (grid_x, grid_y, grid_z)
                            , method = 'nearest')
        data = DataArray(grid_r, coords=[grid_x[:, 0, 0], grid_y[0, :, 0], grid_z[0, 0]])
        return data

    def _grid_2d(self, drange, resolution=(500, 500)):
        r'''Interpolate points in same elevation angle into regular 2-d grid'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            r = self._r_resample(drange=drange)[0]
            phi = self.elevanglelist[self.anglelist_r]
        elif self.radartype == 'CC':
            phin = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            r = list()
            for i in phin:
                r.append(self.reflectivity(i, drange))
            r = np.array(r)
            phi = np.zeros(9)
        elif self.radartype == 'SC':
            phin = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            r = list()
            for i in phin:
                r.append(self.reflectivity(i, drange))
            r = np.array(r)
            phi = np.zeros(9)
        x = list()
        y = list()
        for i in phi:
            self.set_elevation_angle(i)
            coor = self.projection(datatype='et')
            x.append(coor[0])
            y.append(coor[1])
        lon = np.array(x)
        lat = np.array(y)
        x_res, y_res = resolution
        t_x = np.linspace(lon.min(), lon.max(), x_res)
        t_y = np.linspace(lat.min(), lat.max(), y_res)
        x_, y_ = np.meshgrid(t_x, t_y)
        fin = list()
        count = 0
        while count < r.shape[0]:
            grid_r = griddata((lon[count].flatten(), lat[count].flatten()), r[count].flatten(), (x_, y_), method='nearest')
            fin.append(grid_r)
            count += 1
        return x_, y_, np.concatenate(fin).reshape(len(phi), resolution[0], resolution[1])

    @check_radartype(['SA', 'SB', 'CA', 'CB', 'CC', 'SC'])
    def composite_reflectivity(self, drange=230):
        r'''Find max ref value in single coordinate and mask data outside
        obervation range'''
        lon, lat, r_raw = self._grid_2d(drange=drange)
        r_max = np.max(r_raw, axis=0)
        xdim = r_max.shape[0]
        xcoor = np.linspace(-1 * drange, drange, xdim)
        x, y = np.meshgrid(xcoor, xcoor)
        dist = np.sqrt(np.abs(x ** 2 + y ** 2))
        return lon, lat, np.ma.array(r_max, mask=(dist > drange))


class DPRadar:
    r'''Class handling dual-polarized radar reading and plotting.'''
    def __init__(self, filepath):
        from metpy.io.nexrad import Level2File
        self.f = Level2File(filepath)
        dtime = self.f.dt
        self.timestr = dtime.strftime('%Y%m%d%H%M%S')
        self.name = self.f.stid.decode()
        self.el = np.array([ray[0][0].el_angle for ray in self.f.sweeps])
        self.stationlon = self.f.sweeps[0][0][1].lon
        self.stationlat = self.f.sweeps[0][0][1].lat

    def get_data(self, level, drange, dtype):
        if dtype.__class__ is str:
            self.dtype = dtype.upper().encode()
        elif dtype.__class__ is bytes:
            self.dtype = dtype.upper()
        if self.dtype in [b'REF', b'VEL', b'SW', b'ZDR', b'PHI', b'RHO']:
            if self.dtype in [b'ZDR', b'PHI', b'RHO'] and level in [1, 3]:
                level -= 1
                warnings.warn('Elevation angle {} does not contain {} data, automatically switch to level {}'.format(
                    level + 1, self.dtype.decode(), level))
            hdr = self.f.sweeps[level][0][4][self.dtype][0]
            self.reso = hdr.gate_width
            raw = np.array([ray[4][self.dtype][1] for ray in self.f.sweeps[level]])
        elif self.dtype == b'KDP':
            phi = np.array([ray[4][b'PHI'][1] for ray in self.f.sweeps[level]])
            header = self.f.sweeps[0][0][4][b'PHI'][0]
            self.reso = header.gate_width
            kdp = list()
            for i in phi:
                i_ = np.append(i[1:], i[-1])
                kdp.append((i_ - i) / (self.reso * 2))
            raw = np.concatenate(kdp).reshape(phi.shape)
        else:
            raise RadarError('Unknown data type {}'.format(self.dtype.decode()))
        cut = raw.T[:int(drange / self.reso)]
        self.level = level
        self.drange = drange
        self.elev = self.el[level]
        return cut.T

    def _get_coordinate(self, distance, angle):
        if self.stationlat is None or self.stationlon is None:
            raise RadarError('The position of radar should be assigned before projection')
        if self.elev is None:
            raise RadarError('The elevation angle is not defined')
        elev = self.elev
        deltav = np.cos(angle[:, np.newaxis]) * distance * np.cos(np.deg2rad(elev))
        deltah = np.sin(angle[:, np.newaxis]) * distance * np.cos(np.deg2rad(elev))
        deltalat = deltav / 111
        actuallat = deltalat + self.stationlat
        deltalon = deltah / 111
        actuallon = deltalon + self.stationlon
        return actuallon, actuallat

    @staticmethod
    def _height(distance, elevation):
        return distance * np.sin(elevation * deg2rad) + distance ** 2 / (2 * Rm1)

    def projection(self):
        if self.dtype == b'KDP':
            dt = b'PHI'
        else:
            dt = self.dtype
        header = self.f.sweeps[self.level][0][4][dt][0]
        gatenum = header.num_gates
        firstgate = header.first_gate
        data_range = np.arange(gatenum) * self.reso + firstgate
        azi = np.array([ray[self.level].az_angle for ray in self.f.sweeps[self.level]]) * deg2rad
        datalength = int(self.drange / self.reso)
        lonx, latx = self._get_coordinate(data_range[:datalength], azi)
        height = self._height(data_range[:datalength], self.elev) * np.ones(azi.shape[0])[:, np.newaxis]
        return lonx, latx, height

    def draw_ppi(self, level, drange, dtype, draw_author=True, smooth=False, dpi=350, draw_china=True):
        suffix = ''
        data = self.get_data(level, drange, dtype)
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        coor = self.projection()
        lons = coor[0]
        lats = coor[1]
        lonm, latm = np.max(lons), np.max(lats)
        x_delta = lonm - self.stationlon
        y_delta = latm - self.stationlat
        angle_offset = np.cos(self.elev * deg2rad)
        x_offset = x_delta / angle_offset
        y_offset = y_delta / angle_offset
        plt.style.use('dark_background')
        m = Basemap(llcrnrlon=self.stationlon - x_offset, urcrnrlon=self.stationlon + x_offset
                , llcrnrlat=self.stationlat - y_offset, urcrnrlat=self.stationlat + y_offset, resolution="l")
        if self.dtype.decode() == 'REF':
            typestring = 'Base Reflectivity'
            cmaps = r_cmap
            cbar_cmap = r_cmap
            norms = cmx.Normalize(0, 75)
            norms_ = norms
            if smooth:
                m.contourf(lons.flatten(), lats.flatten(), data.flatten(), 256, cmap=r_cmap_smooth
                           , norm=norms, tri=True)
                suffix = '_smooth'
            else:
                data[data <= 2] = None
        elif self.dtype.decode() == 'ZDR':
            typestring = 'Differential Ref.'
            cmaps = zdr_cmap
            cbar_cmap = zdr_cbar
            norms = cmx.Normalize(-4, 5)
            norms_ = cmx.Normalize(0, 1)
            tlabel = ['', '5', '4', '3.5', '3', '2.5', '2', '1.5', '1', '0.8', '0.5', '0.2', '0'
                      , '-1', '-2', '-3', '-4']
        elif self.dtype.decode() == 'KDP':
            typestring = 'Specific Diff. Phase'
            cmaps = kdp_cmap
            cbar_cmap = kdp_cbar
            norms = cmx.Normalize(-0.8, 20)
            norms_ = cmx.Normalize(0, 1)
            data[data < -10] = np.nan
            data[data >50] = np.nan
            tlabel = ['', '20', '7', '3.1', '2.4', '1.7', '1.1', '0.75', '0.5', '0.33', '0.22', '0.15'
                      , '0.1', '-0.1', '-0.2', '-0.4', '-0.8']
        elif self.dtype.decode() == 'RHO':
            typestring = 'Correlation Coe.'
            cmaps = cc_cmap
            cbar_cmap = cc_cbar
            norms = cmx.Normalize(0, 0.99)
            norms_ = cmx.Normalize(0, 1)
            tlabel = ['', '0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '0.92', '0.9', '0.85', '0.8'
                      , '0.7', '0.6', '0.5', '0.3', '0.1', '0']
        m.pcolormesh(lons, lats, data, norm=norms, cmap=cmaps)
        r1 = data[np.logical_not(np.isnan(data))]
        if draw_china:
            m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.5, color='grey')
            m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=0.8, color='white')
        else:
            m.resolution = 'f'
            m.drawstates(linewidth=0.8, color='white')
            m.drawcoastlines(linewidth=0.8, color='white')
        plt.axis('off')
        ax2 = fig.add_axes([0.92, 0.12, 0.04, 0.35])
        cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cbar_cmap, norm=norms_, orientation='vertical', drawedges=False)
        cbar.ax.tick_params(labelsize=8)
        if self.dtype.decode() in ['ZDR', 'KDP', 'RHO']:
            cbar.set_ticks(np.linspace(0, 1, 17))
            cbar.set_ticklabels(tlabel)
        ax2.text(0, 2.13, typestring, fontproperties=font2)
        ax2.text(0, 2.09, 'Range: {:.0f}km'.format(self.drange), fontproperties=font2)
        ax2.text(0, 2.05, 'Resolution: {:.2f}km'.format(self.reso) , fontproperties=font2)
        ax2.text(0, 2.01, 'Date: {}.{}.{}'.format(self.timestr[:4], self.timestr[4:6], self.timestr[6:8]), fontproperties=font2)
        ax2.text(0, 1.97, 'Time: {}:{}'.format(self.timestr[8:10], self.timestr[10:12]), fontproperties=font2)
        ax2.text(0, 1.93, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.89, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.85, 'Elev: {:.2f}deg'.format(self.elev), fontproperties=font2)
        ax2.text(0, 1.81, 'Max: {:.1f}'.format(np.max(r1)), fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.73, 'Made by HCl', fontproperties=font2)
        plt.savefig('{}{}_{}_{:.1f}_{}_{}{}.png'.format(
            folderpath, self.name, self.timestr, self.elev, self.drange, self.dtype.decode().upper(), suffix)
                    , bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig
