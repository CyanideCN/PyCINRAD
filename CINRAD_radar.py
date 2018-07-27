# -*- coding: utf-8 -*-
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cmx
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
from scipy.interpolate import griddata
import os
import warnings
import datetime
from form_colormap import form_colormap

mpl.rc('font', family='Arial')
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
con = (180 / 4096) * 0.125
Rm1 = 8500
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

class RadarError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class CINRAD():
    def __init__(self, filepath, radartype=None):
        self.level = None
        self.drange = None
        self.stationlon = None
        self.stationlat = None
        self.radarheight = None
        self.elev = None
        self.name = None
        self.all_info = None
        self.code = None
        filename, filetype = os.path.splitext(filepath)
        filename = filename.split('/')[-1]
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
        if radartype == ('SA' or 'SB'):
            blocklength = 2432
        elif radartype == 'CB':
            blocklength = 4132
        elif radartype == 'CC':
            blocklength = 3000
        f = open(filepath, 'rb')
        vraw = list()
        rraw = list()
        copy = f.read(blocklength)
        f.seek(0)
        datalength = len(f.read())
        num = int(datalength / blocklength)
        if radartype == ('SA' or 'SB' or 'CB'):
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
                if radartype == ('SA' or 'SB'):
                    R = np.fromstring(a[128:588], dtype='u1')
                    V = np.fromstring(a[128:1508], dtype='u1')
                elif radartype == 'CB':
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
            self.rad = np.deg2rad(self.aziangle)
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
                a = f.read(3000)
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

    def set_radar_height(self, height):
        self.radarheight = height

    def set_elevation_angle(self, angle):
        self.elev = angle

    def _get_radar_info(self):
        r'''Get radar station info from the station database according to the station code.'''
        if self.code is None:
            warnings.warn('Radar code undefined')
            return None
        pos = np.where(radarinfo[0] == self.code)[0]
        name = radarinfo[1][pos][0]
        lon = radarinfo[2][pos][0]
        lat = radarinfo[3][pos][0]
        radartype = radarinfo[4][pos][0]
        radarheight = radarinfo[5][pos][0]
        return name, lon, lat, radartype, radarheight

    def _update_radar_info(self):
        r'''Update radar station info automatically.'''
        info = self._get_radar_info()
        if info is None:
            warnings.warn('Auto fill radar station info failed, '+
                          'use set_code and then _update_radarinfo manually instead.')
        else:
            self.set_station_position(info[1], info[2])
            self.set_station_name(info[0])
            self.set_radar_height(info[4])

    def _height(self, distance, elevation):
        return distance * np.sin(elevation) + distance ** 2 / (2 * Rm1) + self.radarheight / 1000

    def _azimuth_position(self, azimuth):
        r'''Find the relative position of a certain azimuth angle in the data array.'''
        count = 0
        azim = self.aziangle[self.boundary[self.level]:self.boundary[self.level + 1]]
        azim_r = azim.tolist()
        azim_r.reverse()
        while count < len(azim):
            count += 1
            if azimuth > azim[0]:
                if (azimuth - azim[count]) * (azimuth - azim[count + 1]) < 0:
                    print(azim[count])
                    return count
            else:
                if (azimuth - azim_r[count]) * (azimuth - azim_r[count + 1]) < 0:
                    print(azim[len(azim) - count - 1])
                    return len(azim) - count - 1

    def reflectivity(self, level, drange):
        r'''Clip desired range of reflectivity data.'''
        if self.radartype == ('SA' or 'SB' or 'CB'):
            print(self.z[self.boundary[level]])
            self.elev = self.z[self.boundary[level]]
        self.level = level
        self.drange = drange
        length = self.rraw.shape[1] * self.Rreso
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.')
            self.drange = int(self.rraw.shape[1] * self.Rreso)
        if self.radartype == ('SA' or 'SB' or 'CB'):
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
        num = 0
        while num < len(radialavr) - 1:
            delta = radialavr[num + 1] - radialavr[num]
            if delta > 20 and num> 50:
                break
            num += 1
        rm = r1[:num]
        nanmatrix = np.zeros((int(drange / self.Rreso) - num, r1.shape[1]))# * np.nan
        r1 = np.concatenate((rm, nanmatrix))
        return r1.transpose()

    def velocity(self, level, drange):
        r'''Clip desired range of velocity data.'''
        print(self.z[self.boundary[level]])
        self.elev = self.z[self.boundary[level]]
        self.drange = drange
        self.level = level
        length = self.vraw.shape[1] * self.Vreso
        if self.vraw.shape[1] * self.Vreso < drange:
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
        deltalon = deltah / 111#(111 * np.cos(np.deg2rad(actuallat)))
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
        return np.array(latx), np.array(lonx), np.array(height)

    def projection(self, datatype='r'):
        r'''Calculate the geographic coordinates of the requested data range.'''
        if self.radartype == ('SA' or 'SB' or 'CB'):
            length = self.boundary[self.level + 1] - self.boundary[self.level]
        elif self.radartype == 'CC':
            length = 512
        count = 0
        if datatype == 'r':
            r = np.arange(self.Rreso, int(self.drange) + self.Rreso, self.Rreso)
            xshape, yshape = (length, int(self.drange / self.Rreso))
            if self.radartype == ('SA' or 'SB' or 'CB'):
                theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype == 'CC':
                theta = np.deg2rad(np.linspace(0, 360, length))
        elif datatype == 'v':
            r = np.arange(self.Vreso, int(self.drange) + self.Vreso, self.Vreso)
            xshape, yshape = (length, int(self.drange / self.Vreso))
            if self.radartype == ('SA' or 'SB' or 'CB'):
                theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
            elif self.radartype == 'CC':
                theta = np.deg2rad(np.linspace(0, 360, length))
        elif datatype == 'et':
            r = np.arange(1, 231, 1)
            xshape, yshape = (361, 230)
            theta = np.deg2rad(np.arange(0, 361, 1))
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
        angle_offset = np.cos(np.deg2rad(self.elev))
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
            pos = self._azimuth_position(azimuth)
            if pos is None:
                nanarray = np.zeros((drange))
                rhi.append(nanarray.tolist())
            else:
                rhi.append(cac[pos])
            theta = np.deg2rad(self.elev)
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

    def echo_top(self, threshold=18):
        '''Calculate max height of echo data'''
        data = self._r_resample()
        r = np.ma.array(data[0], mask=(data[0] > threshold))
        elev = np.deg2rad(np.delete(self.elevanglelist, [1, 3])).tolist()
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
                    h_pt = h_mask[-1 * k][i][j]#从高仰角开始索引
                    r_pt = data[0][-1 * k][i][j]
                    h_pt_ = hght[-1 * k][i][j]
                    vert_h.append(h_pt)
                    vert_r.append(r_pt)
                    vert_h_.append(h_pt_)
                vertical = np.array(vert_h)
                position = np.where(vertical > 0)[0]
                try:
                    pos = position[0]
                except IndexError:#空array
                    et.append(0)
                    continue
                if pos == 0:#第一个反射率大于18的仰角是最高的仰角
                    height = vertical[pos]
                    et.append(height)
                else:
                    e1 = elev[pos]
                    try:
                        e2 = elev[pos - 1]
                    except IndexError:
                        et.append(vertical[pos])
                        continue
                    z1 = vert_r[pos]
                    z2 = vert_r[pos - 1]
                    h1 = vertical[pos]
                    h2 = vert_h_[pos - 1]
                    w1 = (z1 - threshold) / (z1 - z2)
                    w2 = 1 - w1
                    et.append(w1 * h2 + w2 * h1)#线性内插
        return np.array(et).reshape(361, 230)