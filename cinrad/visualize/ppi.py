# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
import warnings
import json
from typing import Union, Optional, Any, List
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from xarray import Dataset

from cinrad.visualize.utils import *
from cinrad.utils import MODULE_DIR
from cinrad.projection import get_coordinate
from cinrad.io.level3 import StormTrackInfo
from cinrad._typing import Number_T
from cinrad.common import get_dtype, is_radial
from cinrad.visualize.layout import *
from cartopy.io.shapereader import Reader


__all__ = ["PPI"]


def update_dict(d1: dict, d2: dict):
    r"""
    Update the content of the first dict with entries in the second,
    and return the copy.
    """
    d = d1.copy()
    for k, v in d2.items():
        d[k] = v
    return d


def opposite_color(c):
    r"""
    Return the opposite color in white & black.
    """
    if c == "black":
        return "white"
    elif c == "white":
        return "black"


class PPI(object):
    r"""
    Create a figure plotting plan position indicator

    By default, norm, cmap, and colorbar labels will be determined by the
    data type.

    Args:
        data (xarray.Dataset): The data to be plotted.

        fig (matplotlib.figure.Figure): The figure to plot on. Optional.

        norm (matplotlib.colors.Normalize): Customized normalize object. Optional.

        cmap (matplotlib.colors.Colormap): Customized colormap. Optional.

        nlabel (int): Number of labels on the colorbar, will only be used when label is
         also passed. Optional.

        label (list[str]): Colorbar labels. Optional.

        dpi (int): DPI of the figure. Default 350.

        highlight (str, list[str]): Areas to be highlighted. Optional.

        coastline (bool): Plot coastline on the figure if set to True. Default False.

        extent (list[float]): The extent of figure. Optional.

        add_city_names (bool): Label city names on the figure if set to True. Default False.

        plot_labels (bool): Text scan information on the side of the plot. Default True.

        text_param (dict): Optional parameters passed to matplotlib text function.

        add_shps (bool): Add shape files to the figure. Default True.
    """

    # The CRS of data is believed to be PlateCarree.
    # i.e., the coordinates are longitude and latitude.
    data_crs = ccrs.PlateCarree()

    def __init__(
        self,
        data: Dataset,
        fig: Optional[Any] = None,
        norm: Optional[Any] = None,
        cmap: Optional[Any] = None,
        nlabel: int = 10,
        label: Optional[List[str]] = None,
        dpi: Number_T = 350,
        highlight: Optional[Union[str, List[str]]] = None,
        coastline: bool = False,
        extent: Optional[List[Number_T]] = None,
        section: Optional[Dataset] = None,
        style: str = "black",
        add_city_names: bool = False,
        plot_labels: bool = True,
        text_param: Optional[dict] = None,
        add_shps: bool = True,
        **kwargs
    ):
        self.data = data
        self.dtype = get_dtype(data)
        self.settings = {
            "cmap": cmap,
            "norm": norm,
            "nlabel": nlabel,
            "label": label,
            "highlight": highlight,
            "coastline": coastline,
            "path_customize": False,
            "extent": extent,
            "slice": section,
            "style": style,
            "add_city_names": add_city_names,
            "plot_labels": plot_labels,
            "is_inline": is_inline(),
            "add_shps": add_shps,
        }
        if fig is None:
            if style == "transparent":
                self.fig = plt.figure(figsize=FIG_SIZE_TRANSPARENT, dpi=dpi)
            else:
                self.fig = plt.figure(figsize=FIG_SIZE, dpi=dpi)
                self.fig.patch.set_facecolor(style)
            plt.axis("off")
        else:
            self.fig = fig
        # avoid in-place modification
        self.text_pos = TEXT_AXES_POS.copy()
        self.cbar_pos = CBAR_POS.copy()
        self.font_kw = default_font_kw.copy()
        self.font_kw["color"] = opposite_color(style)
        if text_param:
            # Override use input setting
            self.font_kw = update_dict(self.font_kw, text_param)
        self._plot_ctx = dict()
        self.rf_flag = "RF" in data
        self._fig_init = False
        self._plot(**kwargs)
        if is_inline():
            # In inline mode, figure will not be dynamically changed
            # call this action at initialization
            self._text_before_save()

    def __call__(self, fpath):
        ext_name = fpath.split(".")
        if len(ext_name) > 1:
            all_fmt = self.fig.canvas.get_supported_filetypes()
            if ext_name[-1] in all_fmt:
                self.settings["path_customize"] = True
        else:
            if not fpath.endswith(os.path.sep):
                fpath += os.path.sep
        return self._save(fpath)

    def _norm(self):
        if self.settings["norm"]:
            n = self.settings["norm"]
            if self.settings["label"]:
                clabel = self.settings["label"]
            else:
                nlabel = self.settings["nlabel"]
                clabel = np.linspace(n.vmin, n.vmax, nlabel).astype(str)
            return n, n, clabel
        else:
            n = norm_plot[self.dtype]
            n2 = norm_cbar[self.dtype]
            return n, n2, cbar_text[self.dtype]

    def _cmap(self):
        if self.settings["cmap"]:
            c = self.settings["cmap"]
            return c, c
        else:
            c = cmap_plot[self.dtype]
            c2 = cmap_cbar[self.dtype]
            return c, c2

    def _plot(self, **kwargs):
        lon = self.data["longitude"].values
        lat = self.data["latitude"].values
        var = self.data[self.dtype].values
        extent = self.settings["extent"]
        if not extent:
            extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        self.settings["extent"] = extent
        # When plot single radar, azimuthal equidistant projection is used.
        # The data which has code like 'ZXXXX' is considered as single radar.
        code = self.data.site_code
        if is_radial(self.data) or (code.startswith("Z") and len(code) == 5):
            proj = ccrs.AzimuthalEquidistant(
                central_longitude=self.data.site_longitude,
                central_latitude=self.data.site_latitude,
            )
        else:
            proj = ccrs.PlateCarree()
        self.geoax: GeoAxes = create_geoaxes(
            self.fig, proj, extent=extent, style=self.settings["style"]
        )
        if self.settings["style"] == "black":
            self.geoax.patch.set_facecolor("black")
        self._plot_ctx["var"] = var
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        self.geoax.pcolormesh(
            lon,
            lat,
            var,
            norm=pnorm,
            cmap=pcmap,
            transform=self.data_crs,
            shading="auto",
            **kwargs
        )
        if self.rf_flag:
            rf = self.data["RF"].values
            self.geoax.pcolormesh(
                lon,
                lat,
                rf,
                norm=norm_plot["RF"],
                cmap=cmap_plot["RF"],
                transform=self.data_crs,
                shading="auto",
                **kwargs
            )
        if not self.settings["extent"]:
            self._autoscale()
        if self.settings["add_shps"]:
            add_shp(
                self.geoax,
                proj,
                coastline=self.settings["coastline"],
                style=self.settings["style"],
                extent=self.geoax.get_extent(self.data_crs),
            )
        if self.settings["highlight"]:
            draw_highlight_area(self.settings["highlight"])
        if self.settings["add_city_names"]:
            self._add_city_names()

        if self.settings["slice"]:
            self.plot_cross_section(self.settings["slice"])
        self._fig_init = True

    def _text(self):
        def _draw(ax: Any, y_index: int, text: str):
            """
            Draw text on the axes.
            """
            y = INIT_TEXT_POS - TEXT_SPACING * y_index
            ax.text(0, y, text, **self.font_kw)

        # axes used for text which has the same x-position as
        # the colorbar axes (for matplotlib 3 compatibility)
        var = self._plot_ctx["var"]
        ax2 = self.fig.add_axes(self.text_pos)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.yaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        # Make VCP21 the default scanning strategy
        task = self.data.attrs.get("task", "VCP21")
        if self.data.tangential_reso >= 0.1:
            reso = "{:.2f}km".format(self.data.tangential_reso)
        else:
            reso = "{:.0f}m".format(self.data.tangential_reso * 1000)
        s_time = datetime.strptime(self.data.scan_time, "%Y-%m-%d %H:%M:%S")
        texts = [
            prodname[self.dtype],
            "Range: {:.0f}km".format(self.data.range),
            "Resolution: {}".format(reso),
            "Date: {}".format(s_time.strftime("%Y.%m.%d")),
            "Time: {}".format(s_time.strftime("%H:%M")),
            "RDA: " + (self.data.site_name or "Unknown"),
            "Task: {}".format(task),
            "Elev: {:.2f}deg".format(self.data.elevation),
            "Max: {:.1f}{}".format(np.nanmax(var), unit[self.dtype]),
        ]
        if self.dtype.startswith("VEL"):
            min_vel = "Min: {:.1f}{}".format(np.nanmin(var), unit[self.dtype])
            texts.append(min_vel)
        for i, text in enumerate(texts):
            _draw(ax2, i, text)

    def _text_before_save(self):
        # Finalize texting here
        if self.settings["style"] == "transparent":
            return
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        if self.settings["plot_labels"]:
            self._text()
        cax = self.fig.add_axes(self.cbar_pos)
        cbar = ColorbarBase(
            cax, cmap=ccmap, norm=cnorm, orientation="vertical", drawedges=False
        )
        cbar.ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelsize=10,
            colors=self.font_kw["color"],
        )
        cbar.outline.set_visible(False)
        if not isinstance(clabel, type(None)):
            cbar.set_ticks(np.linspace(cnorm.vmin, cnorm.vmax, len(clabel)))
            cbar.set_ticklabels(clabel, **self.font_kw)

    def _save(self, fpath: str):
        if not self.settings["is_inline"]:
            self._text_before_save()
        if not self.settings["path_customize"]:
            if not fpath.endswith(os.path.sep):
                fpath += os.path.sep
            if self.settings["slice"]:
                data = self.settings["slice"]
                sec = "_{:.2f}_{:.2f}_{:.2f}_{:.2f}".format(
                    data.start_lat, data.start_lon, data.end_lat, data.end_lon
                )
            else:
                sec = ""
            path_string = "{}{}_{}_{:.1f}_{}_{}{}.png".format(
                fpath,
                self.data.site_code,
                datetime.strptime(self.data.scan_time, "%Y-%m-%d %H:%M:%S").strftime(
                    "%Y%m%d%H%M%S"
                ),
                self.data.elevation,
                self.data.range,
                self.dtype.upper(),
                sec,
            )
        else:
            path_string = fpath
        save_options = dict(pad_inches=0)
        if self.settings["style"] == "transparent":
            save_options["transparent"] = True
        else:
            if self.settings["style"] == "white":
                save_options["facecolor"] = "white"
            elif self.settings["style"] == "black":
                save_options["facecolor"] = "black"
        plt.savefig(path_string, **save_options)
        # plt.close("all")
        return path_string

    def plot_range_rings(
        self,
        _range: Union[int, float, list],
        color: str = "white",
        linewidth: Number_T = 0.5,
        **kwargs
    ):
        r"""Plot range rings on PPI plot."""
        slon, slat = self.data.site_longitude, self.data.site_latitude
        if isinstance(_range, (int, float)):
            _range = [_range]
        theta = np.linspace(0, 2 * np.pi, 800)
        for d in _range:
            x, y = get_coordinate(d, theta, 0, slon, slat, h_offset=False)
            # self.ax.plot(x, y, color=color, linewidth=linewidth, **kwargs)
            self.geoax.plot(
                x,
                y,
                color=color,
                linewidth=linewidth,
                transform=self.data_crs,
                **kwargs
            )

    def plot_ring_rays(
        self,
        angle: Union[int, float, list],
        range: int,
        color: str = "white",
        linewidth: Number_T = 0.5,
        **kwargs
    ):
        r"""Plot ring rays on PPI plot."""
        slon, slat = self.data.site_longitude, self.data.site_latitude
        if isinstance(angle, (int, float)):
            angle = [angle]
        for a in angle:
            theta = np.deg2rad(a)
            x, y = get_coordinate(range, theta, 0, slon, slat, h_offset=False)
            self.geoax.plot(
                [slon, x],
                [slat, y],
                color=color,
                linewidth=linewidth,
                transform=self.data_crs,
                **kwargs
            )

    def add_custom_shp(
        self,
        shp_path: str,
        encoding: str = "gbk",
        color: str = "white",
        linewidth: Number_T = 0.5,
        **kwargs
    ):
        """
        Add custom shapefile to the plot.
        """
        reader = Reader(shp_path, encoding=encoding)
        self.geoax.add_geometries(
            geoms=list(reader.geometries()),
            crs=ccrs.PlateCarree(),
            edgecolor=color,
            facecolor="None",
            zorder=3,
            linewidth=linewidth,
            **kwargs
        )

    def plot_cross_section(
        self,
        data: Dataset,
        ymax: Optional[int] = None,
        linecolor: Optional[str] = "white",
        interpolate: bool = True,
    ):
        # May add check to ensure the data is slice data
        r"""Plot cross section data below the PPI plot."""
        self.settings["slice"] = data
        # The axes to plot c-section is below the main axes
        # the height of it is a quarter of the height of main axes
        # so the positions of the figure, the main axes, the colorbar axes
        # should be adjusted accordingly.
        # TODO: remove hardcode and calculate positions automatically
        self.fig.set_size_inches(10, 10)
        self.geoax.set_position([0, 0.2, 0.8, 0.8])
        ax2 = self.fig.add_axes([0, 0.01, 0.8, 0.17])
        ax2.patch.set_facecolor(self.settings["style"])
        # transform coordinates
        self.text_pos[1] = self.text_pos[1] * 0.8 + 0.2
        self.text_pos[3] = self.text_pos[3] * 0.8
        self.cbar_pos[1] = self.cbar_pos[1] * 0.8 + 0.2
        self.cbar_pos[3] = self.cbar_pos[3] * 0.8
        ax2.yaxis.set_ticks_position("right")
        ax2.set_xticks([])
        dtype = get_dtype(data)
        sl = data[dtype].values
        if dtype == "REF":
            # visualization improvement for reflectivity
            sl[np.isnan(sl)] = -0.1
        xcor = data["x_cor"]
        ycor = data["y_cor"]
        cmap = sec_plot[dtype]
        norm = norm_plot[dtype]
        if interpolate:
            ax2.contourf(xcor, ycor, sl, 256, cmap=cmap, norm=norm)
        else:
            ax2.pcolormesh(xcor, ycor, sl, cmap=cmap, norm=norm, shading="auto")
        if ymax:
            ax2.set_ylim(0, ymax)
        else:
            ax2.set_ylim(0, 15)
        ax2.set_title(
            "Start: {}N {}E".format(data.start_lat, data.start_lon)
            + " End: {}N {}E".format(data.end_lat, data.end_lon),
            **self.font_kw
        )
        self.geoax.plot(
            [data.start_lon, data.end_lon],
            [data.start_lat, data.end_lat],
            marker="x",
            color=linecolor,
            transform=self.data_crs,
            zorder=5,
        )
        for spine in ax2.spines.values():
            spine.set_color(self.font_kw["color"])
        ax2.tick_params(colors=self.font_kw["color"])

    def storm_track_info(self, filepath: str):
        r"""
        Add storm tracks from Nexrad Level III (PUP) STI product file
        """
        sti = StormTrackInfo(filepath)
        if len(sti.info.keys()) == 0:
            warnings.warn("No storm track to plot", RuntimeWarning)
            return
        else:
            stlist = sti.storm_list
            # extent = self.geoax.get_extent()
            for st in stlist:
                past = sti.track(st, "past")
                fcs = sti.track(st, "forecast")
                current = sti.current(st)
                if past:
                    self.geoax.plot(
                        *past,
                        marker=".",
                        color="white",
                        zorder=4,
                        markersize=5,
                        transform=self.data_crs
                    )
                if fcs:
                    self.geoax.plot(
                        *fcs,
                        marker="+",
                        color="white",
                        zorder=4,
                        markersize=5,
                        transform=self.data_crs
                    )
                self.geoax.scatter(
                    *current,
                    marker="o",
                    s=15,
                    zorder=5,
                    color="lightgrey",
                    transform=self.data_crs
                )
                # if (current[0] > extent[0]) and (current[0] < extent[1]) and (current[1] > extent[2]) and (current[1] < extent[3]):
                #    self.geoax.text(current[0] - 0.03, current[1] - 0.03, st, color='white', zorder=4)

    def gridlines(self, draw_labels: bool = True, linewidth: Number_T = 0, **kwargs):
        r"""Draw grid lines on cartopy axes"""
        from cartopy import __version__

        if not isinstance(self.geoax.projection, ccrs.PlateCarree):
            # Some workaround about the issue that cartopy version lower than 0.18 cannot
            # draw ticks on AzimuthalEquidistant plot
            if __version__ < "0.18":
                warnings.warn(
                    "Cartopy older than 0.18 cannot draw ticks on AzimuthalEquidistant plot.",
                    RuntimeWarning,
                )
                return
        liner = self.geoax.gridlines(
            draw_labels=draw_labels,
            linewidth=linewidth,
            transform=self.data_crs,
            rotate_labels=False,
            xlabel_style={"color": self.font_kw["color"]},
            ylabel_style={"color": self.font_kw["color"]},
            **kwargs
        )
        liner.top_labels = False
        liner.right_labels = False
        if __version__ >= "0.20":
            liner.ypadding = -5
            liner.xpadding = -5

    def _add_city_names(self):
        with open(
            os.path.join(MODULE_DIR, "data", "chinaCity.json"), encoding="utf-8"
        ) as j:
            js = json.load(j)
        name = np.concatenate([[j["name"] for j in i["children"]] for i in js])
        lon = np.concatenate([[j["log"] for j in i["children"]] for i in js]).astype(
            float
        )
        lat = np.concatenate([[j["lat"] for j in i["children"]] for i in js]).astype(
            float
        )
        extent = self.settings["extent"]
        fraction = (extent[1] - extent[0]) * 0.04
        target_city = (
            (lon > (extent[0] + fraction))
            & (lon < (extent[1] - fraction))
            & (lat > (extent[2] + fraction))
            & (lat < (extent[3] - fraction))
        )
        for nm, stlon, stlat in zip(
            name[target_city], lon[target_city], lat[target_city]
        ):
            self.geoax.text(
                stlon,
                stlat,
                nm,
                **{**self.font_kw, "color": "darkgrey"},
                color="darkgrey",
                transform=self.data_crs,
                horizontalalignment="center",
                verticalalignment="center"
            )

    def _autoscale(self):
        llon, ulon, llat, ulat = self.geoax.get_extent()
        lon_delta = ulon - llon
        lat_delta = ulat - llat
        if lon_delta == lat_delta:
            return
        if lon_delta > lat_delta:
            # The long axis is x-axis
            lat_center = (ulat + llat) / 2
            lat_extend = lon_delta / 2
            llat = lat_center - lat_extend
            ulat = lat_center + lat_extend
        elif lon_delta < lat_delta:
            # The long axis is y-axis
            lon_center = (ulon + llon) / 2
            lon_extend = lat_delta / 2
            llon = lon_center - lon_extend
            ulon = lon_center + lon_extend
        self.geoax.set_extent([llon, ulon, llat, ulat], self.geoax.projection)
