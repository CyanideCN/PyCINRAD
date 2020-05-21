# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from xarray import Dataset

from cinrad.datastruct import Slice_
from cinrad.common import get_dtype
from cinrad.visualize.utils import sec_plot, norm_plot, prodname, save

__all__ = ["Section", "RHI"]


class Section(object):
    def __init__(
        self,
        data: Dataset,
        hlim: int = 15,
        interpolate: bool = True,
        figsize: tuple = (10, 5),
    ):
        self.data = data
        self.dtype = get_dtype(data)
        self.settings = {
            "hlim": hlim,
            "interp": interpolate,
            "figsize": figsize,
        }
        self.rhi_flag = hasattr(data.attrs, "azimuth")
        self._plot()

    def _plot(self):
        from cinrad.visualize.utils import plot_kw

        rhi = self.data[self.dtype]
        xcor = self.data["x_cor"]
        ycor = self.data["y_cor"]
        rmax = np.nanmax(rhi.values)
        plt.figure(figsize=self.settings["figsize"], dpi=300)
        plt.grid(
            True, linewidth=0.50, linestyle="-.", color="white"
        )  ## 修改于2019-01-22 By WU Fulang
        cmap = sec_plot[self.dtype]
        norm = norm_plot[self.dtype]
        if self.settings["interp"]:
            plt.contourf(
                xcor, ycor, rhi, 128, cmap=cmap, norm=norm,
            )
        else:
            plt.pcolormesh(
                xcor, ycor, rhi, cmap=cmap, norm=norm,
            )
        plt.ylim(0, self.settings["hlim"])
        if self.rhi_flag:
            title = "Range-Height Indicator\n"
        else:
            title = "Vertical cross-section ({})\n".format(prodname[self.dtype])
        title += "Station: {} ".format(self.data.site_name)
        if self.rhi_flag:
            # RHI scan type
            title += "Range: {:.0f}km Azimuth: {:.0f}° ".format(
                self.data.range, self.data.azimuth
            )
        else:
            title += "Start: {}N {}E ".format(self.data.start_lat, self.data.start_lon)
            title += "End: {}N {}E ".format(self.data.end_lat, self.data.end_lon)
        title += "Time: " + self.data.scan_time.strftime("%Y.%m.%d %H:%M ")
        title += "Max: {:.1f}".format(rmax)
        plt.title(title, **plot_kw)
        lat_pos = np.linspace(self.data.start_lat, self.data.end_lat, 6)
        lon_pos = np.linspace(self.data.start_lon, self.data.end_lon, 6)
        tick_formatter = lambda x, y: "{:.2f}N\n{:.2f}E".format(x, y)
        ticks = list(map(tick_formatter, lat_pos, lon_pos))
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ticks)
        plt.ylabel("Height (km)", **plot_kw)  ## 修改于2019-01-22 By WU Fulang


    def __call__(self, fpath: str):
        path_string = "{}_{}_VCS_{}N{}E_{}N{}E.png".format(
            self.data.site_code,
            self.data.scan_time.strftime("%Y%m%d%H%M%S"),
            self.data.start_lat,
            self.data.start_lon,
            self.data.end_lat,
            self.data.end_lon,
        )
        if os.path.isdir(fpath):
            save_path = os.path.join(fpath, path_string)
        else:
            save_path = fpath
        save(save_path)


class XXXSection(object):
    def __init__(
        self,
        data: Slice_,
        hlim: int = 15,
        interpolate: bool = True,
        figsize: tuple = (10, 5),
    ):
        self.data = data
        self.dtype = data.dtype
        self.settings = {"hlim": hlim, "interp": interpolate, "figsize": figsize}
        self.path_customize = False

    def _plot(self, fpath: str):
        from cinrad.visualize.utils import plot_kw

        rhi = self.data.data
        xcor = self.data.xcor
        ycor = self.data.ycor
        rmax = rhi.max()
        plt.style.use("dark_background")
        plt.figure(figsize=self.settings["figsize"], dpi=300)
        plt.grid(
            True, linewidth=0.50, linestyle="-.", color="white"
        )  ## 修改于2019-01-22 By WU Fulang
        cmap = sec_plot[self.data.dtype]
        norm = norm_plot[self.data.dtype]
        if self.settings["interp"]:
            plt.contourf(
                xcor, ycor, rhi, 128, cmap=cmap, norm=norm,
            )
        else:
            plt.pcolormesh(
                xcor, ycor, rhi, cmap=cmap, norm=norm,
            )
        plt.ylim(0, self.settings["hlim"])
        stps = self.data.geoinfo["stp_s"]
        enps = self.data.geoinfo["enp_s"]
        stp = self.data.geoinfo["stp"]
        enp = self.data.geoinfo["enp"]
        plt.title(
            "Vertical cross-section ({})\nStation: {} Start: {} End: {} Time: {} Max: {:.1f}".format(
                prodname[self.dtype],
                self.data.name,
                stps,
                enps,
                self.data.scantime.strftime("%Y.%m.%d %H:%M"),
                rmax,
            ),
            **plot_kw
        )
        lat_pos = np.linspace(stp[1], enp[1], 6)
        lon_pos = np.linspace(stp[0], enp[0], 6)
        tick_formatter = lambda x, y: "{:.2f}N\n{:.2f}E".format(x, y)
        ticks = list(map(tick_formatter, lat_pos, lon_pos))
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ticks)
        plt.ylabel("Height (km)", **plot_kw)  ## 修改于2019-01-22 By WU Fulang
        if self.path_customize:
            path_string = fpath
        else:
            path_string = "{}{}_{}_VCS_{}N{}E_{}N{}E.png".format(
                fpath,
                self.data.code,
                self.data.scantime.strftime("%Y%m%d%H%M%S"),
                stp[1],
                stp[0],
                enp[1],
                enp[0],
            )
        plt.savefig(path_string, bbox_inches="tight")

    def __call__(self, *fpath):
        if not fpath:
            fpath = os.path.join(str(Path.home()), "PyCINRAD")
        else:
            fpath = fpath[0]
            if fpath.upper().endswith(".PNG"):
                self.path_customize = True
            else:
                if not fpath.endswith(os.path.sep):
                    fpath += os.path.sep
        return self._plot(fpath)


class RHI(object):
    def __init__(self, data: Slice_, hlim: int = 15, interpolate: bool = True):
        self.data = data
        self.dtype = data.dtype
        self.hlim = hlim
        self.interp = interpolate
        self.azimuth = data.geoinfo["azimuth"]
        self.path_customize = False

    def _plot(self, fpath: str):
        from cinrad.visualize.utils import plot_kw

        rhi = self.data.data
        xcor = self.data.xcor
        ycor = self.data.ycor
        rmax = rhi.max()
        plt.style.use("dark_background")
        plt.figure(figsize=(10, 5), dpi=300)
        plt.grid(True, linewidth=0.5, linestyle="-.", color="white")
        norm = norm_plot[self.data.dtype]
        cmap = sec_plot[self.data.dtype]
        if interpolate:
            plt.contourf(
                xcor, ycor, rhi, 128, cmap=cmap, norm=norm,
            )
        else:
            plt.pcolormesh(
                xcor, ycor, rhi, cmap=cmap, norm=norm,
            )
        plt.ylim(0, self.hlim)
        plt.title(
            "Range-Height Indicator\nStation: {} Data: {} Range: {:.0f}km Azimuth: {:.0f}° Time: {}".format(
                self.data.name,
                self.data.dtype,
                self.data.xcor.max(),
                self.azimuth,
                self.data.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            )
        )
        plt.ylabel("Height (km)", **plot_kw)
        if self.path_customize:
            path_string = fpath
        else:
            path_string = "{}{}_{}_RHI_{:.0f}_{:.0f}_{}.png".format(
                fpath,
                self.data.code,
                self.data.scantime.strftime("%Y%m%d%H%M%S"),
                self.azimuth,
                self.data.xcor.max(),
                self.dtype,
            )
        plt.savefig(path_string, bbox_inches="tight")

    def __call__(self, *fpath):
        if not fpath:
            fpath = os.path.join(str(Path.home()), "PyCINRAD")
        else:
            fpath = fpath[0]
            if fpath.upper().endswith(".PNG"):
                self.path_customize = True
            else:
                if not fpath.endswith(os.path.sep):
                    fpath += os.path.sep
        return self._plot(fpath)
