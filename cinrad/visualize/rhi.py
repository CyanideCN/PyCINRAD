# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from xarray import Dataset

from cinrad.common import get_dtype
from cinrad.visualize.utils import sec_plot, norm_plot, prodname, default_font_kw
from cinrad.visualize.ppi import opposite_color, update_dict

__all__ = ["Section"]


class Section(object):
    def __init__(
        self,
        data: Dataset,
        hlim: int = 15,
        interpolate: bool = True,
        figsize: tuple = (10, 5),
        style: str = "black",
        text_param: dict = None,
    ):
        # TODO: Use context manager to control style
        self.data = data
        self.dtype = get_dtype(data)
        self.settings = {
            "hlim": hlim,
            "interp": interpolate,
            "figsize": figsize,
            "style": style,
        }
        self.font_kw = default_font_kw.copy()
        self.font_kw["color"] = opposite_color(style)
        if text_param:
            # Override use input setting
            self.font_kw = update_dict(self.font_kw, text_param)
        self.rhi_flag = "azimuth" in data.attrs
        self._plot()

    def _plot(self):

        rhi = self.data[self.dtype]
        xcor = self.data["x_cor"]
        ycor = self.data["y_cor"]
        rmax = np.nanmax(rhi.values)
        fig = plt.figure(figsize=self.settings["figsize"], dpi=300)
        ax = plt.gca()
        fig.patch.set_facecolor(self.settings["style"])
        ax.set_facecolor(self.settings["style"])
        plt.grid(
            True, linewidth=0.50, linestyle="-.", color=self.font_kw["color"]
        )  ## 修改于2019-01-22 By WU Fulang
        cmap = sec_plot[self.dtype]
        norm = norm_plot[self.dtype]
        if self.settings["interp"]:
            plt.contourf(
                xcor,
                ycor,
                rhi,
                128,
                cmap=cmap,
                norm=norm,
            )
        else:
            plt.pcolormesh(xcor, ycor, rhi, cmap=cmap, norm=norm, shading="auto")
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
        title += "Time: " + datetime.strptime(
            self.data.scan_time, "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y.%m.%d %H:%M ")
        title += "Max: {:.1f}".format(rmax)
        plt.title(title, **self.font_kw)
        lat_pos = np.linspace(self.data.start_lat, self.data.end_lat, 6)
        lon_pos = np.linspace(self.data.start_lon, self.data.end_lon, 6)
        tick_formatter = lambda x, y: "{:.2f}N\n{:.2f}E".format(x, y)
        ticks = list(map(tick_formatter, lat_pos, lon_pos))
        cor_max = xcor.values.max()
        plt.xticks(
            np.array([0, 0.2, 0.4, 0.6, 0.8, 1]) * cor_max, ticks, **self.font_kw
        )
        plt.ylabel("Height (km)", **self.font_kw)  ## 修改于2019-01-22 By WU Fulang
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = plt.colorbar(sm, ax=ax)
        for spine in ax.spines.values():
            spine.set_color(self.font_kw["color"])
        ax.tick_params(colors=self.font_kw["color"])
        cb.ax.tick_params(colors=self.font_kw["color"])

    def __call__(self, fpath: str):
        if os.path.isdir(fpath):
            if self.rhi_flag:
                path_string = "{}{}_{}_RHI_{:.0f}_{:.0f}_{}.png".format(
                    fpath,
                    self.data.site_code,
                    datetime.strptime(
                        self.data.scan_time, "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y%m%d%H%M%S"),
                    self.data.azimuth,
                    self.data.range,
                    self.dtype,
                )
            else:
                path_string = "{}_{}_VCS_{}N{}E_{}N{}E.png".format(
                    self.data.site_code,
                    datetime.strptime(
                        self.data.scan_time, "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y%m%d%H%M%S"),
                    self.data.start_lat,
                    self.data.start_lon,
                    self.data.end_lat,
                    self.data.end_lon,
                )
            save_path = os.path.join(fpath, path_string)
        else:
            save_path = fpath
        plt.savefig(save_path, bbox_inches="tight")
        return save_path
