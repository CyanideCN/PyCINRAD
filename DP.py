#coding=utf-8
import numpy as np
from metpy.io.nexrad import Level2File
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.font_manager import FontProperties
import matplotlib.colors as cmx
from CINRAD_radar import RadarError, Rm1, deg2rad
from form_colormap import form_colormap

mpl.rc('font', family='Arial')
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\Dengl.ttf")
font2 = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\msyh.ttc")
folderpath = 'D:\\'
r_cmap = form_colormap('colormap\\radarnmc.txt', sep=True)
r_cmap_smooth = form_colormap('colormap\\radarnmca.txt', sep=False, spacing='v')
norm1 = cmx.Normalize(0, 75)

class DPRadar:
    def __init__(self, filepath):
        self.f = Level2File(filepath)
        dtime = self.f.dt
        self.timestr = dtime.strftime('%Y%m%d%H%M%S')
        self.name = self.f.stid.decode()
        self.el = np.array([ray[0][0].el_angle for ray in self.f.sweeps])
        self.stationlon = self.f.sweeps[0][0][1].lon
        self.stationlat = self.f.sweeps[0][0][1].lat
        self.level = None
        self.drange = None
        self.dtype = None
        self.reso = None

    def get_ref(self, level, drange):
        self.level = level
        self.drange = drange
        self.dtype = b'REF'
        self.elev = self.el[level]
        ref_hdr = self.f.sweeps[level][0][4][b'REF'][0]
        raw_ref = np.array([ray[4][b'REF'][1] for ray in self.f.sweeps[level]])
        ref_cut = raw_ref.T[:int(drange / ref_hdr.gate_width)]
        return ref_cut.T

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

    def projection(self):
        header = self.f.sweeps[self.level][0][4][self.dtype][0]
        self.reso = header.gate_width
        gatenum = header.num_gates
        firstgate = header.first_gate
        data_range = np.arange(gatenum) * self.reso + firstgate
        azi = np.array([ray[self.level].az_angle for ray in self.f.sweeps[self.level]]) * deg2rad
        datalength = int(self.drange / self.reso)
        latx = list()
        lonx = list()
        height = list()
        count = 0
        while count < len(azi):
            for i in data_range[:datalength]:
                t = azi[count]
                lons, lats = self._get_coordinate(i, t)
                h = i * np.sin(np.deg2rad(self.elev)) + i ** 2 / (2 * Rm1 ** 2)
                latx.append(lats)
                lonx.append(lons)
                height.append(h)
            count = count + 1
        xshape, yshape = (720, len(latx) // 720)
        x, y, z = np.array(lonx), np.array(latx), np.array(height)
        return x.reshape(xshape, yshape), y.reshape(xshape, yshape), z.reshape(xshape, yshape)

    def draw_ref(self, level, drange, draw_author=True, smooth=False, dpi=350):
        suffix = ''
        data = self.get_ref(level, drange)
        fig = plt.figure(figsize=(10, 10), dpi=dpi)
        lons, lats, hgh = self.projection()
        lonm, latm = np.max(lons), np.max(lats)
        x_delta = lonm - self.stationlon
        y_delta = latm - self.stationlat
        angle_offset = np.cos(self.elev * deg2rad)
        x_offset = x_delta / angle_offset
        y_offset = y_delta / angle_offset
        plt.style.use('dark_background')
        m = Basemap(llcrnrlon=self.stationlon - x_offset, urcrnrlon=self.stationlon + x_offset
                , llcrnrlat=self.stationlat - y_offset, urcrnrlat=self.stationlat + y_offset, resolution="l")
        typestring = 'Base Reflectivity'
        cmaps = r_cmap
        norms = norm1
        r1 = data[np.logical_not(np.isnan(data))]
        if smooth:
            m.contourf(lons.flatten(), lats.flatten(), data.flatten(), 256, cmap=r_cmap_smooth, norm=norms, tri=True)
            suffix = '_smooth'
        else:
            data[data <= 2] = None
            m.pcolormesh(lons, lats, data, norm=norms, cmap=cmaps)
        m.readshapefile('shapefile\\City', 'states', drawbounds=True, linewidth=0.5, color='grey')
        m.readshapefile('shapefile\\Province', 'states', drawbounds=True, linewidth=0.8, color='white')
        plt.axis('off')
        ax2 = fig.add_axes([0.92, 0.12, 0.04, 0.35])
        cbar = mpl.colorbar.ColorbarBase(ax2, cmap=cmaps, norm=norms, orientation='vertical', drawedges=False)
        cbar.ax.tick_params(labelsize=8)
        ax2.text(0, 2.13, typestring, fontproperties=font2)
        ax2.text(0, 2.09, 'Range: {:.0f}km'.format(self.drange), fontproperties=font2)
        ax2.text(0, 2.05, 'Resolution: {:.2f}km'.format(self.reso) , fontproperties=font2)
        ax2.text(0, 2.01, 'Date: {}.{}.{}'.format(self.timestr[:4], self.timestr[4:6], self.timestr[6:8]), fontproperties=font2)
        ax2.text(0, 1.97, 'Time: {}:{}'.format(self.timestr[8:10], self.timestr[10:12]), fontproperties=font2)
        ax2.text(0, 1.93, 'RDA: ' + self.name, fontproperties=font2)
        ax2.text(0, 1.89, 'Mode: Precipitation', fontproperties=font2)
        ax2.text(0, 1.85, 'Elev: {:.2f}deg'.format(self.elev), fontproperties=font2)
        ax2.text(0, 1.81, 'Max: {:.1f}dBz'.format(np.max(r1)), fontproperties=font2)
        if draw_author:
            ax2.text(0, 1.73, 'Made by HCl', fontproperties=font2)
        plt.savefig('{}{}_{:.1f}_{}_{}{}.png'.format(
            folderpath, self.timestr, self.elev, self.drange, self.dtype.decode().upper(), suffix), bbox_inches='tight', pad_inches = 0)
        plt.cla()
        del fig