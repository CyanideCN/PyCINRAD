# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cmx
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
from scipy.interpolate import griddata
import warnings
import datetime
from pathlib import Path
from form_colormap import form_colormap

mpl.rc('font', family='Arial')
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
con = (180 / 4096) * 0.125
Rm1 = 8500
deg2rad = np.pi / 180
folderpath = 'D:\\'

r_cmap = form_colormap('colormap\\radarnmc.txt', sep=True)
v_cmap = form_colormap('colormap\\radarnmc2.txt', sep=False)
vel_cbar = form_colormap('colormap\\radarnmc2a.txt', sep=True)
rhi_cmap_smooth = form_colormap('colormap\\radarnmc.txt', sep=False, spacing='v')
r_cmap_smooth = form_colormap('colormap\\radarnmca.txt', sep=False, spacing='v')
et_cmap = form_colormap('colormap\\et.txt', sep=False)
et_cbar = form_colormap('colormap\\etbar.txt', sep=True)
radarinfo = np.load('RadarStation.npy')
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)

def check_radartype(accept_list):
    def check(func):
        def inner(self, *args, **kwargs):
            if self.radartype not in accept_list:
                raise RadarError('{} radar is not supported for fuction {}'.format(self.radartype, func.__name__))
            return func(self, *args, **kwargs)
        return inner
    return check

def find_intersection(line1, line2):
    k1 = line1[0]
    b1 = line1[1]
    k2 = line2[0]
    b2 = line2[1]
    if k1 == k2:
        raise ValueError('No intersection exist')
    x = (b2 - b1) / (k1 - k2)
    y = k1 * x + b1
    return (x, y)

class RadarError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class Radar():
    def __init__(self, filepath, radartype=None):
        self.level = None
        self.drange = None
        self.stationlon = None
        self.stationlat = None
        self.radarheight = None
        self.elev = None
        self.name = None
        self.code = None
        self.azim = None
        path = Path(filepath)
        filename = path.name
        filetype = path.suffix
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
        self.radartype = radartype
        if radartype in ['SA', 'SB']:
            blocklength = 2432
        elif radartype in ['CA', 'CB']:
            blocklength = 4132
        elif radartype == 'CC':
            blocklength = 3000
        elif radartype in ['SC', 'CD']:
            blocklength = 4000
        else:
            raise RadarError('Radar type should be specified')
        if filetype.endswith('bz2'):
            import bz2
            f = bz2.open(filepath, 'rb')
        else:
            f = open(filepath, 'rb')
        vraw = list()
        rraw = list()
        copy = f.read(blocklength)
        f.seek(0)
        datalength = len(f.read())
        num = int(datalength / blocklength)
        if radartype in ['SA', 'SB', 'CA', 'CB']:
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
                if radartype in ['SA', 'SB']:
                    R = np.fromstring(a[128:588], dtype='u1')
                    V = np.fromstring(a[128:1508], dtype='u1')
                elif radartype in ['CA', 'CB']:
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
            self.eleang = np.array(eleang)
            self.vraw = np.array(vraw)
            self.dv = veloreso[0]
            anglelist = np.arange(0, anglenum[0], 1)
            self.anglelist_r = np.delete(anglelist, [1, 3])
            self.anglelist_v = np.delete(anglelist, [0, 2])
            self.elevanglelist = self.z[self.boundary][:-1]
        elif radartype == 'CC':
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
        elif radartype in ['SC', 'CD']:
            utc_offset = datetime.timedelta(hours=8)
            f.seek(853)
            scantime = datetime.datetime(year=np.fromstring(f.read(2), 'u2')[0], month=np.fromstring(f.read(1), 'u1')[0],
                                         day=np.fromstring(f.read(1), 'u1')[0], hour=np.fromstring(f.read(1), 'u1')[0],
                                         minute=np.fromstring(f.read(1), 'u1')[0], second=np.fromstring(f.read(1), 'u1')[0]) - utc_offset
            f.seek(1024)
            count = 0
            while count < num:
                a = f.read(blocklength)
                r = np.fromstring(a[:1000], dtype=np.short).astype(float)
        self.timestr = scantime.strftime('%Y%m%d%H%M%S')
        self._update_radar_info()
        
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
            warnings.warn('Radar code undefined')
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
            warnings.warn('Auto fill radar station info failed, please set code manually')
        else:
            self.set_station_position(info[1], info[2])
            self.set_station_name(info[0])
            self.set_radar_height(info[4])

    def _height(self, distance, elevation):
        return distance * np.sin(elevation) + distance ** 2 / (2 * Rm1) + self.radarheight / 1000

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

    @check_radartype(['SA', 'SB', 'CB', 'CC'])
    def reflectivity(self, level, drange):
        r'''Clip desired range of reflectivity data.'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            self.elev = self.z[self.boundary[level]]
            print(self.elev)
            if level in [1, 3]:
                warnings.warn('Use this elevation angle may yield unexpected result.')
        self.level = level
        self.drange = drange
        length = self.rraw.shape[1] * self.Rreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.')
            self.drange = int(self.rraw.shape[1] * self.Rreso)
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            dbz = (self.rraw - 2) / 2 - 32
            r = dbz[self.boundary[level]:self.boundary[level + 1]]
            r1 = r.transpose()[:int(drange / self.Rreso)]
        elif self.radartype == 'CC':
            dbz = self.rraw / 10
            r1 = dbz[level * 512:(level + 1) * 512, :int(drange / self.Rreso)].transpose()
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
        return r1.transpose()

    @check_radartype(['SA', 'SB', 'CB'])
    def velocity(self, level, drange):
        r'''Clip desired range of velocity data.'''
        if level in [0, 2]:
            warnings.warn('Use this elevation angle may yield unexpected result.')
        self.elev = self.z[self.boundary[level]]
        print(self.elev)
        self.drange = drange
        self.level = level
        length = self.vraw.shape[1] * self.Vreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.')
            self.drange = int(self.vraw.shape[1] * self.Vreso)
        if self.dv == 2:
            v = (self.vraw - 2) / 2 - 63.5
        elif self.dv == 4:
            v = (self.vraw - 2) - 127
        v = v[self.boundary[level]:self.boundary[level + 1]]
        v1 = v.transpose()[:int(drange / self.Vreso)]
        v1[v1 == -64.5] = np.nan
        rf = np.ma.array(v1, mask=(v1 != -64))
        return v1.transpose(), rf.transpose()

    def _get_coordinate(self, distance, angle):
        r'''Convert polar coordinates to geographic coordinates with the given radar station position.'''
        if self.stationlat is None or self.stationlon is None:
            raise RadarError('The position of radar should be assigned before projection')
        if self.elev is None:
            raise RadarError('The elevation angle is not defined')
        elev = self.elev
        deltav = np.cos(angle) * distance * np.cos(np.deg2rad(elev))
        deltah = np.sin(angle) * distance * np.cos(np.deg2rad(elev))
        deltalat = deltav / 111
        actuallat = deltalat + self.stationlat
        deltalon = deltah / 111
        actuallon = deltalon + self.stationlon
        return actuallon, actuallat

    def _polar2cart(self, distance, azimuth):
        latx = list()
        lonx = list()
        height = list()
        count = 0
        while count < len(azimuth):
            for i in distance:
                t = azimuth[count]
                lons, lats = self._get_coordinate(i, t)
                h = i * np.sin(np.deg2rad(self.elev)) + i ** 2 / (2 * Rm1 ** 2)
                latx.append(lats)
                lonx.append(lons)
                height.append(h)
            count = count + 1
        return np.array(lonx), np.array(latx), np.array(height)

    def projection(self, datatype='r'):
        r'''Calculate the geographic coordinates of the requested data range.'''
        if self.radartype in ['SA', 'SB', 'CA', 'CB']:
            length = self.boundary[self.level + 1] - self.boundary[self.level]
        elif self.radartype == 'CC':
            length = 512
        if datatype == 'r':
            r = np.arange(self.Rreso, int(self.drange) + self.Rreso, self.Rreso)
            xshape, yshape = (length, int(self.drange / self.Rreso))
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype == 'CC':
                theta = np.linspace(0, 360, length) * deg2rad
        elif datatype == 'v':
            r = np.arange(self.Vreso, int(self.drange) + self.Vreso, self.Vreso)
            xshape, yshape = (length, int(self.drange / self.Vreso))
            if self.radartype in ['SA', 'SB', 'CA', 'CB']:
                theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype == 'CC':
                theta = np.linspace(0, 360, length) * deg2rad
        elif datatype == 'et':
            r = np.arange(1, 231, 1)
            xshape, yshape = (361, 230)
            theta = np.arange(0, 361, 1) * deg2rad
        x, y, z = self._polar2cart(r, theta)
        lons = x.reshape(xshape, yshape)
        lats = y.reshape(xshape, yshape)
        hgh = z.reshape(xshape, yshape)# + self.radarheight / 1000
        return lons, lats, hgh

    def draw_ppi(self, level, drange, datatype='r', draw_author=True, smooth=False, dpi=350):
        r'''Plot reflectivity PPI scan with the default plot settings.'''
        suffix = ''
        if datatype == 'r':
            data = self.reflectivity(level, drange)
        elif datatype == 'v':
            data, rf = self.velocity(level, drange)
        elif datatype == 'et':
            data = self.echo_top()
            self.set_elevation_angle(0)
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        lons, lats, hgh = self.projection(datatype=datatype)
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
            m.pcolormesh(lons, lats, rf, cmap=rfmap, norm=cmx.Normalize(-1, 0))
        elif datatype == 'et':
            typestring = 'Echo Tops'
            cmaps = et_cbar
            norms = cmx.Normalize(0, 1)
            reso = self.Rreso
            data[data > 25] = 0
            m.pcolormesh(lons, lats, data, cmap=et_cmap, norm=cmx.Normalize(0, 21))
        m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.5, color='grey')
        m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=0.8, color='white')
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
        if datatype == 'r':
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
        if self.name is None:
            raise RadarError('Name of radar is not defined')
        xc, yc, rhi = self.rhi(azimuth, drange, startangle=startangle, stopangle=stopangle
                               , height=height, interpolation=interpolation)
        rmax = np.round_(np.max(rhi[np.logical_not(np.isnan(rhi))]), 1)
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 4), dpi=200)
        plt.contourf(xc, yc, rhi, 128, cmap=rhi_cmap_smooth, norm=norm1, corner_mask=False)
        plt.ylim(0, height)
        plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                  self.name, azimuth, self.timestr[:4], self.timestr[4:6], self.timestr[6:8], self.timestr[8:10]
                  , self.timestr[10:12], rmax), fontproperties=font2)
        plt.ylabel('Altitude (km)')
        plt.xlabel('Range (km)')
        plt.savefig('{}{}_{}_RHI_{}_{}.png'.format(folderpath, self.code, self.timestr, self.drange, azimuth)
                    , bbox_inches='tight')

    def _r_resample(self):
        Rrange = np.arange(1, 231, 1)
        Trange = np.arange(0, 361, 1)
        dist, theta = np.meshgrid(Rrange, Trange)
        r_resampled = list()
        for i in self.anglelist_r:
            r = self.reflectivity(i, 230)
            azimuth = self.aziangle[self.boundary[i]:self.boundary[i + 1]]
            dist_, theta_ = np.meshgrid(Rrange, azimuth)
            r_ = griddata((dist_.flatten(), theta_.flatten()), r.flatten(), (dist, theta), method='nearest')
            r_resampled.append(r_)
        r_res = np.concatenate(r_resampled)
        return r_res.reshape(r_res.shape[0] // 361, 361, 230), dist, theta

    @check_radartype(['SA', 'SB'])
    def echo_top(self, threshold=18):
        '''Calculate max height of echo data'''
        data = self._r_resample()
        r = np.ma.array(data[0], mask=(data[0] > threshold))
        elev = np.delete(self.elevanglelist * deg2rad, [1, 3]).tolist()
        h_ = list()
        for i in elev:
            h = self._height(data[1], i)
            h_.append(h)
        hght = np.concatenate(h_).reshape(r.shape)
        h_mask = hght * r.mask
        et = list()
        for i in range(0, 361):
            for j in range(0, 230):
                vert_h = list()
                vert_r = list()
                vert_h_ = list()
                for k in range(1, 10):
                    h_pt = h_mask[-1 * k][i][j]#index from highest angle
                    r_pt = data[0][-1 * k][i][j]
                    h_pt_ = hght[-1 * k][i][j]
                    vert_h.append(h_pt)
                    vert_r.append(r_pt)
                    vert_h_.append(h_pt_)
                vertical = np.array(vert_h)
                position = np.where(vertical > 0)[0]
                try:
                    pos = position[0]
                except IndexError:#empty array
                    et.append(0)
                    continue
                if pos == 0:
                    height = vertical[pos]
                    et.append(height)
                else:
                    try:
                        elev[pos - 1]
                    except IndexError:
                        et.append(vertical[pos])
                        continue
                    z1 = vert_r[pos]
                    z2 = vert_r[pos - 1]
                    h1 = vertical[pos]
                    h2 = vert_h_[pos - 1]
                    w1 = (z1 - threshold) / (z1 - z2)
                    w2 = 1 - w1
                    et.append(w1 * h2 + w2 * h1)#linear interpolation
        return np.array(et).reshape(361, 230)

    @check_radartype(['SA'])
    def cross_section(self, startpos, endpos):
        s_dis = startpos[0]
        s_ang = startpos[1]
        e_dis = endpos[0]
        e_ang = endpos[1]
        raw_delta = abs(s_ang - e_ang)
        if raw_delta > 180:
            delta_ang = 360 - raw_delta
        else:
            delta_ang = raw_delta
        #resolve polar coordinate into x and y axes
        s_x = s_dis * np.sin(s_ang * deg2rad)
        s_y = s_dis * np.cos(s_ang * deg2rad)
        e_x = e_dis * np.sin(e_ang * deg2rad)
        e_y = e_dis * np.cos(e_ang * deg2rad)
        #solve the equation for line
        line_grad = (e_y - s_y) / (e_x - s_x)
        line_intcpt = e_y - line_grad * e_x
        ref = list()
        range_ = list()
        height = list()
        for ag in self.anglelist_r:
            self.set_level(ag)
            s_ang_pos = self._find_azimuth_position(s_ang)
            e_ang_pos = self._find_azimuth_position(e_ang)
            if s_ang < e_ang:
                if e_ang_pos < s_ang_pos:
                    dir_array = np.concatenate((self.azim[s_ang_pos:], self.azim[:e_ang_pos]))
                else:
                    dir_array = self.azim[s_ang_pos:e_ang_pos]
            else:
                if e_ang_pos > s_ang_pos:
                    dir_array = np.concatenate((self.azim[e_ang_pos:], self.azim[:s_ang_pos]))
                else:
                    dir_array = self.azim[s_ang_pos:e_ang_pos]
            normalize = list()
            for i in dir_array:
                if i <= np.pi:
                    normalize.append(np.pi / 2 - i)
                elif i > np.pi and i <= np.pi * 1.5:
                    normalize.append(i - np.pi)
                else:
                    normalize.append(2 * np.pi - i)
            line_grads = np.tan(normalize)
            point_x = list()
            point_y = list()
            for i in line_grads:
                coor = find_intersection((line_grad, line_intcpt), (i, 0))
                point_x.append(coor[0])
                point_y.append(coor[1])
            x = np.array(point_x)
            y = np.array(point_y)
            distance = np.sqrt(x ** 2 + y ** 2)
            angle = np.arctan(x / y)
            count = 0
            r = self.reflectivity(ag, 230)
            r_ = list()
            d_ = list()
            h_ = list()
            theta = self.elev * deg2rad
            while count < len(distance):
                pos1 = int(distance[count] / self.Rreso)
                pos2 = self._find_azimuth_position(dir_array[count] / deg2rad)
                r_.append(r[pos2][pos1])
                d_.append(distance[count] * np.cos(theta))
                h_.append(distance[count] * np.sin(theta))
                count += 1
            ref.append(r_)
            range_.append(d_)
            height.append(h_)
        return np.concatenate(range_), np.concatenate(height), np.concatenate(ref)