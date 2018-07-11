# -*- coding: utf-8 -*-
from form_colormap import form_colormap
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as cmx
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
import os
import warnings
import datetime

mpl.rc('font', family='Arial')
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
con = (180 / 4096) * 0.125
IR = 1.21
RE = 6371
folderpath = 'D:\\Meteorology\\Matplotlib\\Basemap\\'

nmcradar = form_colormap('colormap\\radarnmc.txt', sep=True)
nmcradar2 = form_colormap('colormap\\radarnmc2.txt', sep=False)
velcbar = form_colormap('colormap\\radarnmc2a.txt', sep=True)
nmcradarc = form_colormap('colormap\\radarnmc.txt', sep=False, spacing='v')
nmcradarc1 = form_colormap('colormap\\radarnmca.txt', sep=False, spacing='v')
radarinfo = np.load('RadarStation.npy')
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)

class RadarError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class CINRAD():
    def __init__(self, filepath):
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
        elif filename.endswith('A'):
            spart = filename.split('.')
            self.code = None
            radartype ='SA'
        else:
            warnings.warn('Unrecognized filename, please update radar station info manually')
        f = open(filepath, 'rb')
        g = open(filepath, 'rb')
        azimuthx = list()
        eleang = list()
        vraw = list()
        rraw = list()
        blur = list()
        self.boundary = list()
        count = 0
        if radartype == 'SA' or 'SB':
            blocklength = 2432
        elif radartype == 'CB':
            blocklength = 4132
        copy = g.read()
        deltdays = np.fromstring(copy[32:34], dtype='u2')[0]
        deltsecs = np.fromstring(copy[28:32], dtype='u4')[0]
        start = datetime.datetime(1970, 1, 1)
        deltday = datetime.timedelta(days=int(deltdays))
        deltsec = datetime.timedelta(milliseconds=int(deltsecs))
        scantime = start + deltday + deltsec
        self.timestr = scantime.strftime('%Y%m%d%H%M%S')
        self.Rreso = np.fromstring(copy[50:52], dtype='u2')[0] / 1000
        self.Vreso = np.fromstring(copy[52:54], dtype='u2')[0] / 1000
        datalength = len(copy)
        num = int(datalength / blocklength)
        while count < num:
            a = f.read(blocklength)
            blurdist = np.fromstring(a[34:36], dtype='u2')
            azimuth = np.fromstring(a[36:38], dtype='u2')
            datacon = np.fromstring(a[40:42], dtype='u2')
            elevangle = np.fromstring(a[42:44], dtype='u2')
            anglenum = np.fromstring(a[44:46], dtype='u2')
            refdist = np.fromstring(a[46:48], dtype='u2')
            veloreso = np.fromstring(a[70:72], dtype='u2')
            if radartype == 'SA' or 'SB':
                R = np.fromstring(a[128:588], dtype='u1')
                V = np.fromstring(a[128:1508], dtype='u1')
            elif radartype == 'CB':
                R = np.fromstring(a[128:928], dtype='u1')
                V = np.fromstring(a[128:2528], dtype='u1')
            azimuthx.append(azimuth[0])
            eleang.append(elevangle[0])
            vraw.append(V.tolist())
            rraw.append(R.tolist())
            blur.append(blurdist[0] / 10)
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
        self.blurdist = np.array(blur)
        self.radartype = radartype
        self.level = None
        self.drange = None
        self.level = None
        self.stationlon = None
        self.stationlat = None
        self.elev = None
        self.name = None
        anglelist = np.arange(0, anglenum[0], 1)
        self.anglelist_r = np.delete(anglelist, [1, 3])
        self.anglelist_v = np.delete(anglelist, [0, 2])
        self.elevanglelist = self.z[self.boundary]
        self._update_radarinfo()
    
    def set_stationposition(self, stationlon, stationlat):
        self.stationlon = stationlon
        self.stationlat = stationlat

    def set_stationname(self, name):
        self.name = name

    def set_drange(self, drange):
        self.drange = drange

    def set_code(self, code):
        self.code = code

    def _get_radarinfo(self):
        r'''Get radar station info from the station database according to the station code.'''
        if self.code is None:
            warnings.warn('Radar code undefined')
            return None
        pos = np.where(radarinfo[0] == self.code)[0]
        name = radarinfo[1][pos][0]
        lon = radarinfo[2][pos][0]
        lat = radarinfo[3][pos][0]
        radartype = radarinfo[4][pos][0]
        return name, lon, lat, radartype

    def _update_radarinfo(self):
        r'''Update radar station info automatically.'''
        info = self._get_radarinfo()
        if info is None:
            warnings.warn('Auto fill radar station info failed, '+
                          'use set_stationposition and set_stationname manually instead.')
        else:
            self.set_stationposition(info[1], info[2])
            self.set_stationname(info[0])

    def _azimuthposition(self, azimuth):
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

    def reflectivity(self, level, drange, maskdelta=0):
        r'''Clip desired range of reflectivity data.'''
        print(self.z[self.boundary[level]])
        self.elev = self.z[self.boundary[level]]
        self.level = level
        self.drange = drange
        length = self.rraw.shape[1] * self.Rreso
        blur = self.blurdist[self.boundary[level]] - maskdelta
        if length < drange:
            warnings.warn('The input range exceeds maximum range, reset to the maximum range.')
            self.drange = int(self.rraw.shape[1] * self.Rreso)
        dbz = (self.rraw - 2) / 2 - 32
        r = dbz[self.boundary[level]:self.boundary[level + 1]]
        r1 = r.transpose()[:int(drange / self.Rreso)]
        r1[r1 < 0] = 0
        if drange > blur:
            rm = r1[:int(blur / self.Rreso)]
            nanmatrix = np.zeros((int((drange - blur) / self.Rreso), r1.shape[1]))# * np.nan
            r1 = np.concatenate((rm, nanmatrix))
        return r1.transpose()

    def velocity(self, level, drange):
        r'''Clip desired range of velocity data.'''
        print(self.z[self.boundary[level]])
        self.elev = self.z[self.boundary[level]]
        self.drange = drange
        self.level = level
        length = self.vraw.shape[1] * self.Vreso
        blur = self.blurdist[self.boundary[level]]
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
        if drange > blur:
            vm = v1[:int(blur / self.Vreso)]
            nanmatrix = np.zeros((int((drange - blur) / self.Vreso), v1.shape[1]))# * np.nan
            v1 = np.concatenate((vm, nanmatrix))
        rf = np.ma.array(v1, mask=(v1 != -64))
        return v1.transpose(), rf.transpose()

    def getrange(self, stationlon, stationlat):
        r'''Calculate the range of coordinates of the basemap projection.'''
        if self.drange is None:
            raise RadarError('The range of data should be assigned first')
        self.stationlon = stationlon
        self.stationlat = stationlat
        km2lat = 1 / 111
        uplat = stationlat + self.drange * km2lat
        lowlat = stationlat - self.drange * km2lat
        leftlon = stationlon + self.drange / (111 * np.cos(np.deg2rad(stationlat)))
        rightlon = stationlon - self.drange / (111 * np.cos(np.deg2rad(stationlat)))
        return leftlon, rightlon, uplat, lowlat

    def _getcoordinate(self, drange, angle):
        r'''Convert polar coordinates to geographic coordinates with the given radar station position.'''
        if self.drange is None:
            raise RadarError('The range of data should be assigned first')
        if self.stationlat is None or self.stationlon is None:
            raise RadarError('The position of radar should be assigned before projection')
        deltav = np.cos(angle) * drange * np.cos(np.deg2rad(self.eleang[self.boundary[self.level]] * con))
        deltah = np.sin(angle) * drange * np.cos(np.deg2rad(self.eleang[self.boundary[self.level]] * con))
        deltalat = deltav / 111
        actuallat = deltalat + self.stationlat
        deltalon = deltah / (111 * np.cos(np.deg2rad(actuallat)))
        actuallon = deltalon + self.stationlon
        return actuallon, actuallat

    def projection(self, datatype='r'):
        r'''Calculate the geographic coordinates of the requested data range.'''
        length = self.boundary[self.level + 1] - self.boundary[self.level]
        latx = list()
        lonx = list()
        height = list()
        count = 0
        if datatype == 'r':
            r = np.arange(self.Rreso, int(self.drange) + self.Rreso, self.Rreso)
            xshape, yshape = (length, int(self.drange / self.Rreso))
        elif datatype == 'v':
            r = np.arange(self.Vreso, int(self.drange) + self.Vreso, self.Vreso)
            xshape, yshape = (length, int(self.drange / self.Vreso))
        theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
        while count < len(theta):
            for i in r:
                t = theta[count]
                lons, lats = self._getcoordinate(i, t)
                h = i * np.sin(np.deg2rad(self.elev)) + (i * i) / (2 * IR * RE)
                latx.append(lats)
                lonx.append(lons)
                height.append(h)
            count = count + 1
        lons = np.array(lonx).reshape(xshape, yshape)
        lats = np.array(latx).reshape(xshape, yshape)
        hgh = np.array(height).reshape(xshape, yshape)
        return lons, lats, hgh

    def draw_ref(self, level, drange, draw_author=True, smooth=False, dpi=350):
        r'''Plot reflectivity PPI scan with the default plot settings.'''
        suffix = ''
        r = self.reflectivity(level, drange)
        r1 = r[np.logical_not(np.isnan(r))]
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        lonm, loni, latm, lati = self.getrange(self.stationlon, self.stationlat)
        lons, lats, hgh = self.projection()
        plt.style.use('dark_background')
        m = Basemap(llcrnrlon=loni, urcrnrlon=lonm, llcrnrlat=lati, urcrnrlat=latm, resolution="l")
        if smooth:
            m.contourf(lons.flatten(), lats.flatten(), r.flatten(), 128, cmap=nmcradarc1, norm=norm1, tri=True)
            suffix = '_smooth'
        else:
            r[r <= 2] = None
            m.pcolormesh(lons, lats, r, norm=norm1, cmap=nmcradar)
        m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.5, color='grey')
        m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=0.8, color='white')
        plt.axis('off')
        ax2 = fig.add_axes([0.92, 0.17, 0.04, 0.35])
        cbar = mpl.colorbar.ColorbarBase(ax2, cmap=nmcradar, norm=norm1, orientation='vertical', drawedges=False)
        cbar.ax.tick_params(labelsize=8)
        ax2.text(0, 1.84, 'Base Reflectivity', fontproperties=font2)
        ax2.text(0, 1.80, 'Range: {:.0f}km'.format(self.drange), fontproperties=font2)
        ax2.text(0, 1.76, 'Resolution: {:.2f}km'.format(self.Rreso) , fontproperties=font2)
        ax2.text(0, 1.72, 'Date: {}.{}.{}'.format(self.timestr[:4], self.timestr[4:6], self.timestr[6:8]), fontproperties=font2)
        ax2.text(0, 1.68, 'Time: {}:{}'.format(self.timestr[8:10], self.timestr[10:12]), fontproperties=font2)
        ax2.text(0, 1.64, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.60, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.56, 'Elev: {:.2f}deg'.format(self.elev), fontproperties=font2)
        ax2.text(0, 1.48, 'Max: {:.1f}dBz'.format(np.max(r1)), fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.44, 'Made by HCl', fontproperties=font2)
        plt.savefig('{}{}_{}_{:.1f}_{}_R{}.png'.format(folderpath, self.code, self.timestr, self.elev, self.drange, suffix)
                    , bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig

    def rhi(self, azimuth, drange, startangle=0, stopangle=8, height=15, interpolation=False):
        r'''Clip the reflectivity data from certain elevation angles in a single azimuth angle.'''
        rhi = list()
        xcoor = list()
        ycoor = list()
        dist = np.arange(1, drange + 1, 1)
        for i in self.anglelist_r[startangle:stopangle]:
            cac = self.reflectivity(i, drange)
            pos = self._azimuthposition(azimuth)
            if pos is None:
                nanarray = np.zeros((drange))
                rhi.append(nanarray.tolist())
            else:
                rhi.append(cac[pos])
            theta = np.deg2rad(self.elev)
            xcoor.append((dist * np.cos(theta)).tolist())
            ycoor.append((dist * np.sin(theta) + (dist * dist) / (2 * IR * RE)).tolist())
        rhi = np.array(rhi)
        rhi[rhi < 0] = 0
        xc = np.array(xcoor)
        yc = np.array(ycoor)
        if interpolation:
            from metpy import gridding
            warnings.warn('Interpolation takes long time, keep patient')
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
        plt.contourf(xc, yc, rhi, 128, cmap=nmcradarc, norm=norm1, corner_mask=False)
        plt.ylim(0, height)
        plt.title('RHI scan\nStation: {} Azimuth: {}° Time: {}.{}.{} {}:{} Max: {}dbz'.format(
                  self.name, azimuth, self.timestr[:4], self.timestr[4:6], self.timestr[6:8], self.timestr[8:10]
                  , self.timestr[10:12], rmax), fontproperties=font2)
        plt.ylabel('Altitude (km)')
        plt.xlabel('Range (km)')
        #plt.colorbar(cmap=nmcradarc, norm=norm1)
        plt.savefig('{}{}_{}_RHI_{}_{}.png'.format(folderpath, self.code, self.timestr, self.drange, azimuth)
                    , bbox_inches='tight')

    def draw_vel(self, level, drange, draw_author=True, dpi=350):
        r'''Plot velocity PPI scan with the default plot settings.'''
        v, rf = self.velocity(level, drange)
        fig = plt.figure(figsize=(10,10), dpi=dpi)
        lonm, loni, latm, lati = self.getrange(self.stationlon, self.stationlat)
        lons, lats, hgh = self.projection(datatype='v')
        plt.style.use('dark_background')
        m = Basemap(llcrnrlon=loni, urcrnrlon=lonm, llcrnrlat=lati, urcrnrlat=latm, resolution="l")
        m.pcolormesh(lons, lats, v, cmap=nmcradar2, norm=norm2)
        rfmap = cmx.ListedColormap('#660066', '#FFFFFF')
        m.pcolormesh(lons, lats, rf, cmap=rfmap, norm=cmx.Normalize(-1, 0))
        m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.5, color='grey')
        m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=0.8, color='white')
        plt.axis('off')
        ax2 = fig.add_axes([0.92, 0.17, 0.04, 0.35])
        cbar = mpl.colorbar.ColorbarBase(ax2, cmap=velcbar, norm=cmx.Normalize(0, 1), orientation='vertical', drawedges=False)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_ticks(np.linspace(0, 1, 16))
        cbar.set_ticklabels(['RF', '', '27', '20', '15', '10', '5', '1', '0', '-1', '-5', '-10', '-15', '-20', '-27', '-35'])
        ax2.text(0, 1.84, 'Base Velocity', fontproperties=font2)
        ax2.text(0, 1.80, 'Range: {:.0f}km'.format(self.drange), fontproperties=font2)
        ax2.text(0, 1.76, 'Resolution: {:.2f}km'.format(self.Vreso) , fontproperties=font2)
        ax2.text(0, 1.72, 'Date: {}.{}.{}'.format(self.timestr[:4], self.timestr[4:6], self.timestr[6:8]), fontproperties=font2)
        ax2.text(0, 1.68, 'Time: {}:{}'.format(self.timestr[8:10], self.timestr[10:12]), fontproperties=font2)
        ax2.text(0, 1.64, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.60, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.56, 'Elev: {:.2f}deg'.format(self.elev), fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.44, 'Made by HCl', fontproperties=font2)
        plt.savefig('{}{}_{}_{:.1f}_{}_V.png'.format(folderpath, self.code, self.timestr, self.elev, self.drange)
                    , bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig

    def _grid(self, resolution=(200, 200, 10)):
        r'''Convert radar data to grid (test)'''
        from scipy.interpolate import griddata
        datalength = self.boundary[1] - self.boundary[0]
        ref = list()
        lon = list()
        lat = list()
        height = list()
        for i in self.anglelist_r:
            r = self.reflectivity(i, 230)
            x, y, h = self.projection()
            ref.append(r)
            lon.append(x)
            lat.append(y)
            height.append(h.tolist())
        r = np.concatenate(ref)#.reshape(len(self.anglelist), datalength, 230)
        x = np.concatenate(lon)
        y = np.concatenate(lat)
        z = np.concatenate(height)
        x_res, y_res, z_res = resolution
        grid_x, grid_y, grid_z = np.mgrid[np.min(x):np.max(x):x_res * 1j, np.min(y):np.max(y):y_res * 1j
                                          , np.min(z):np.max(z):z_res * 1j]
        grid_r = griddata((x.flatten(), y.flatten(), z.flatten()), r.flatten(), (grid_x, grid_y, grid_z), method = 'nearest')
        return grid_r

    def quickplot(self, radius=230):
        for i in self.anglelist_r:
            self.draw_ref(i, radius, dpi=150, draw_author=True)
        for j in self.anglelist_v:
            self.draw_vel(j, radius, dpi=150, draw_author=True)