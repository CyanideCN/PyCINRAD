# -*- coding: utf-8 -*-
from form_colormap import form_colormap
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cmx
import matplotlib.tri as mtri
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
import datetime
from tkinter import filedialog
import os

mpl.rc('font', family='Arial')
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
con = (180/4096)*0.125
IR = 1.21
RE = 6371
folderpath = 'D:\\Meteorology\\Matplotlib\\Basemap\\'

nmcradar = form_colormap('radarnmc.txt', sep=True)
nmcradar2 = form_colormap('radarnmc2.txt', sep=False)
velcbar = form_colormap('radarnmc2a.txt', sep=True)
nmcradarc = form_colormap('radarnmc.txt', sep=False, spacing='v')
nmcradarc1 = form_colormap('radarnmca.txt', sep=False, spacing='v')
norm1 = cmx.Normalize(0, 75)
norm2 = cmx.Normalize(-35, 27)

PosDict = {'杭州': (120.338, 30.274), 
           '青浦': (120.959, 31.076), 
           '南京': (118.698, 32.191), 
           '合肥': (117.258, 31.868), 
           '蚌埠': (0, 0)}

class RadarError(Exception):
    def __init__(self, description):
        self.dsc = description
    def __str__(self):
        return repr(self.dsc)

class CINRAD():
    def __init__(self, filepath, radartype='SA', version='old'):
        f = open(filepath, 'rb')
        g = open(filepath, 'rb')
        azimuthx = list()
        eleang = list()
        vraw = list()
        rraw = list()
        self.boundary = list()
        count = 0
        if radartype == 'SA' or 'SB':
            blocklength = 2432
            self.Rreso = 1
            self.Vreso = 0.25
        elif radartype == 'CB':
            blocklength = 4132
            self.Rreso = 0.5
            self.Vreso = 0.125
        datalength = len(g.read())
        num = int(datalength / blocklength)
        while count < num:
            a = f.read(blocklength)
            radar = np.fromstring(a[14:16], dtype='u2')
            times = np.fromstring(a[28:32], dtype='u4')
            day = np.fromstring(a[32:34], dtype='u2')
            blurdist = np.fromstring(a[34:36], dtype='u2')
            azimuth = np.fromstring(a[36:38], dtype='u2')
            dataindex = np.fromstring(a[38:40], dtype='u2')
            datacon = np.fromstring(a[40:42], dtype='u2')
            elevangle = np.fromstring(a[42:44], dtype='u2')
            anglenum = np.fromstring(a[44:46], dtype='u2')
            refdist = np.fromstring(a[46:48], dtype='u2')
            dopdist = np.fromstring(a[48:50], dtype='u2')
            refspacing = np.fromstring(a[50:52], dtype='u2')
            dopspacing = np.fromstring(a[52:54], dtype='u2')
            refdistnum = np.fromstring(a[54:56], dtype='u2')
            dopdistnum = np.fromstring(a[56:58], dtype='u2')
            veloreso = np.fromstring(a[70:72], dtype='u2')
            VCPmode = np.fromstring(a[72:74], dtype='u2')
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
            if datacon[0] == 3:
                self.boundary.append(0)
            elif datacon[0] == 0:
                self.boundary.append(count)
            elif datacon[0] == 4:
                self.boundary.append(num - 1)
            count = count+1

        self.rraw = np.array(rraw)
        self.z = np.array(eleang) * con
        self.aziangle = np.array(azimuthx) * con
        self.rad = np.deg2rad(self.aziangle)
        self.eleang = np.array(eleang)
        self.vraw = np.array(vraw)

        self.dv = veloreso[0]
        self.radartype = radartype
        self.version = version
        self.level = None
        self.range = None
        self.level = None
        self.stationlon = None
        self.stationlat = None
        self.elev = None
        self.name = None
        full_name = os.path.split(filepath)[-1]
        self.file_name, self.file_type = os.path.splitext(full_name)
        if version=='old':
            self.timestr = self.file_name.split('_', 9)[4]
        elif version=='new':
            self.timestr = self.file_name + self.file_type[1:-1]
        anglelist = np.arange(0, len(self.boundary)-2, 1)
        self.anglelist = np.delete(anglelist, [1, 3])
    
    def set_stationposition(self, stationlon, stationlat):
        self.stationlon = stationlon
        self.stationlat = stationlat

    def set_stationname(self, name):
        self.name = name

    def set_range(self, range):
        self.range = range

    def _azimuthposition(self, azimuth):
        count = 0
        azim = self.aziangle[self.boundary[self.level]:self.boundary[self.level + 1]]
        if azim[0] < azimuth:
            while count < len(azim):
                count += 1
                if azim[count] - azimuth < azimuth - azim[count - 1] and azim[count] - azimuth <= 1:
                    return count
                elif azim[count] - azimuth > azimuth - azim[count-1] and azimuth - azim[count - 1] <= 1:
                    return count - 1
                else:
                    return None

    def reflectivity(self, level, range):
        print(self.z[self.boundary[level]])
        self.elev = self.z[self.boundary[level]]
        self.range = range
        self.level = level
        dbz = (self.rraw - 2) / 2 - 32
        r = dbz[self.boundary[level]:self.boundary[level + 1]]
        r1 = r.transpose()[:int(range / self.Rreso)]
        r1[r1 < 0] = 0
        return r1.transpose()

    def velocity(self, level, range):
        print(self.z[self.boundary[level]])
        self.elev = self.z[self.boundary[level]]
        self.range = range
        self.level = level
        if self.dv == 2:
            v = (self.vraw - 2) / 2 - 63.5
        elif self.dv == 4:
            v = (self.vraw - 2) -127
        v = v[self.boundary[level]:self.boundary[level + 1]]
        v1 = v.transpose()[:int(range / self.Vreso)]
        v1[v1 == -64.5] = None
        rf = np.ma.array(v1, mask=(v1 != -64))
        return v1.transpose(), rf.transpose()

    def getrange(self, stationlon, stationlat):
        if self.range == None:
            raise RadarError('The range of data should be assigned first')
        self.stationlon = stationlon
        self.stationlat = stationlat
        km2lat = 1 / 111
        uplat = stationlat + self.range * km2lat
        lowlat = stationlat - self.range * km2lat
        leftlon = stationlon + self.range / (111 * np.cos(np.deg2rad(stationlat)))
        rightlon = stationlon - self.range / (111 * np.cos(np.deg2rad(stationlat)))
        return leftlon, rightlon, uplat, lowlat
        
    @staticmethod
    def polar2cart(angle, distance):
        return distance * np.cos(np.deg2rad(angle)), distance * np.sin(np.deg2rad(angle))

    @staticmethod
    def elev2height(angle, distance):
        return distance * np.sin(np.deg2rad(angle))

    def _getcoordinate(self, range, angle):
        if self.range == None:
            raise RadarError('The range of data should be assigned first')
        if self.stationlat == None or self.stationlon == None:
            raise RadarError('The position of radar should be assigned before projection')
        deltav = np.cos(angle) * range * np.cos(np.deg2rad(self.eleang[self.boundary[self.level]] * con))
        deltah = np.sin(angle) * range * np.cos(np.deg2rad(self.eleang[self.boundary[self.level]] * con))
        deltalat = deltav / 111
        actuallat = deltalat + self.stationlat
        deltalon = deltah / (111 * np.cos(np.deg2rad(actuallat)))
        actuallon = deltalon + self.stationlon
        return actuallon, actuallat

    def projection(self, type='r'):
        length = self.boundary[self.level + 1] - self.boundary[self.level]
        latx = list()
        lonx = list()
        height = list()
        count = 0
        if type == 'r':
            r = np.arange(self.Rreso, int(self.range) + self.Rreso, self.Rreso)
            xshape, yshape = (length, int(self.range / self.Rreso))
        elif type == 'v':
            r = np.arange(self.Vreso, int(self.range) + self.Vreso, self.Vreso)
            xshape, yshape = (length, int(self.range / self.Vreso))
        theta = self.rad[self.boundary[self.level]:self.boundary[self.level + 1]]
        while count < len(theta):
            for i in r:
                t = theta[count]
                lons, lats = self._getcoordinate(i, t)
                h = i * np.sin(np.deg2rad(self.elev)) + (i * i)/(2 * IR * RE)
                latx.append(lats)
                lonx.append(lons)
                height.append(h)
            count = count + 1
        lons = np.array(lonx).reshape(xshape, yshape)
        lats = np.array(latx).reshape(xshape, yshape)
        hgh = np.array(height).reshape(xshape, yshape)
        return lons, lats, hgh

    def draw_ref(self, level, range, draw_author=False, smooth=False):
        suffix = ''
        r = self.reflectivity(level, range)
        r1 = r[np.logical_not(np.isnan(r))]
        maxvalue = np.max(r1)
        fig = plt.figure(figsize=(10,10), dpi=350)
        lonm, loni, latm, lati = self.getrange(self.stationlon, self.stationlat)
        lons, lats = self.projection()
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
        ax2.text(0, 1.80, 'Range: ' + str(self.range) + ' km', fontproperties=font2)
        ax2.text(0, 1.76, 'Resolution: 1.00 km', fontproperties=font2)
        ax2.text(0, 1.72, 'Date: ' + self.timestr[:4] + '.' + self.timestr[4:6] + '.'+self.timestr[6:8], fontproperties=font2)
        ax2.text(0, 1.68, 'Time: ' + self.timestr[8:10] + ':' + self.timestr[10:12], fontproperties=font2)
        ax2.text(0, 1.64, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.60, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.56, 'Elev: ' + str(np.round_(self.elev, 2)) + 'deg', fontproperties=font2)
        ax2.text(0, 1.48, 'Max: ' + str(maxvalue) + 'dBz', fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.44, 'Made by HCl', fontproperties=font2)
        if self.version == 'new':
            nameblock = self.file_name + self.file_type[1:-1]
        elif self.version == 'old':
            nameblock = self.file_name
        plt.savefig((folderpath + nameblock + '_' + str(np.round_(self.elev, 1)) 
                     + '_' + str(self.range) + '_R'+ suffix +'.png'), bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig

    def rhi(self, azimuth, range, startangle=0, stopangle=5, height=15, interpolation=False):
        rhi = list()
        xcoor = list()
        ycoor = list()
        dist = np.arange(1, range + 1, 1)
        for i in self.anglelist[startangle:stopangle]:
            cac = self.reflectivity(i, range)
            pos = self._azimuthposition(azimuth)
            if pos == None:
                nanarray = np.zeros((range))
                rhi.append(nanarray.tolist())
            else:
                rhi.append(cac[pos])
            theta = np.deg2rad(self.elev)
            xcoor.append((dist * np.cos(theta)).tolist())
            ycoor.append((dist * np.sin(theta) +(dist * dist)/(2 * IR * RE)).tolist())
        rhi = np.array(rhi)
        rhi[rhi < 0] = 0
        xc = np.array(xcoor)
        yc = np.array(ycoor)
        if interpolation:
            import warnings
            from metpy import gridding
            warnings.warn('Interpolation takes long time, keep patient')
            xi = np.arange(0, range + 1, 1)
            yi = np.arange(0, height + 0.5, 0.5)
            x, y = np.meshgrid(xi, yi)
            z = gridding.natural_neighbor(xc.flatten(), yc.flatten(), rhi.flatten(), x, y)
            return x, y, z
        else:
            return xc, yc, rhi

    def draw_rhi(self, azimuth, range):
        xc, yc, rhi = self.rhi(azimuth, range)
        tri = mtri.Triangulation(xc.flatten(), yc.flatten())
        plt.tricontourf(tri, rhi.flatten(), cmap=nmcradarc, norm=norm1)
        #to be continued
        pass

    def draw_vel(self, level, range, draw_author=False):
        v, rf = self.velocity(level, range)
        fig = plt.figure(figsize=(10,10), dpi=350)
        lonm, loni, latm, lati = self.getrange(self.stationlon, self.stationlat)
        lons, lats = self.projection(type='v')
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
        ax2.text(0, 1.80, 'Range: ' + str(self.range) + ' km', fontproperties=font2)
        ax2.text(0, 1.76, 'Resolution: 0.25 km', fontproperties=font2)
        ax2.text(0, 1.72, 'Date: ' + self.timestr[:4] + '.' + self.timestr[4:6] + '.'+self.timestr[6:8], fontproperties=font2)
        ax2.text(0, 1.68, 'Time: ' + self.timestr[8:10] + ':' + self.timestr[10:12], fontproperties=font2)
        ax2.text(0, 1.64, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.60, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.56, 'Elev: ' + str(np.round_(self.elev, 2)) + 'deg', fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.44, 'Made by HCl', fontproperties=font2)
        if self.version == 'new':
            nameblock = self.file_name + self.file_type[1:-1]
        elif self.version == 'old':
            nameblock = self.file_name
        plt.savefig((folderpath + nameblock + '_' + str(np.round_(self.elev, 1)) 
                        + '_' + str(self.range) + '_V.png'), bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig

    def get_all_info(self):
        datalength = self.boundary[1] - self.boundary[0]
        ref = list()
        lon = list()
        lat = list()
        height = list()
        for i in self.anglelist:
            r = self.reflectivity(i, 230)
            x, y, h = self.projection()
            ref.append(r)
            lon.append(x)
            lat.append(y)
            height.append(h.tolist())
        r = np.concatenate(ref)#.reshape(len(self.anglelist), datalength, 230)
        x = np.concatenate(lon)
        y = np.concatenate(lat)
        h = np.concatenate(height)
        pass
        #to be continued

#file=filedialog.askopenfilename(filetypes=[('BIN Files','*.*')],title='Open CINRAD Data')
file = 'D:\\Meteorology\\雷达基数据\\Z9551_20180518\\2018051807.07A'
radar = CINRAD(file, version='new')
#r = radar.reflectivity(0, 230)

from metpy import gridding
xc, yc, rhi = radar.rhi(230, 230, startangle=0, stopangle=10, interpolation=True)
xi = np.arange(0, 230, 1)
yi = np.arange(0, 15, 0.5)
x, y = np.meshgrid(xi, yi)
z = gridding.natural_neighbor(xc.flatten(), yc.flatten(), rhi.flatten(), x, y)
plt.contourf(xc, yc, rhi, 128, cmap=nmcradarc, norm=norm1, corner_mask=False)

tri = mtri.Triangulation(xc.flatten(), yc.flatten())
plt.tricontourf(tri, rhi.flatten(), 128, cmap=nmcradarc, norm=norm1)
plt.triplot(tri, linewidth=0.3, color='black')
plt.ylim(0, 15)
plt.show()
'''
refiner = mtri.UniformTriRefiner(tri)
tri_refi, z_test_refi = refiner.refine_field(rhi.flatten(), subdiv=2)
plt.tricontourf(tri_refi, z_test_refi, 128, cmap=nmcradarc, norm=norm1)
plt.ylim(0, 15)
plt.show()
'''
#v, rf = radar.velocity(1, 130)

radar.set_stationname('Qingpu')
radar.set_stationposition(PosDict['青浦'][0], PosDict['青浦'][1])
radar.draw_ref(0, 120, draw_author=True, smooth=True)
