# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
from pathlib import Path
import warnings
import json
from typing import Union, Optional, Any, List

import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from xarray import Dataset

from cinrad.visualize.utils import *
from cinrad.projection import get_coordinate
from cinrad.error import RadarPlotError
from cinrad.io.level3 import StormTrackInfo
from cinrad._typing import Number_T
from cinrad.common import get_dtype, is_radial
from cinrad.visualize.layout import TEXT_AXES_POS, TEXT_SPACING, INIT_TEXT_POS, CBAR_POS

__all__ = ["PPI"]


class PPI(object):
    r"""
    Create a figure plotting plan position indicator

    Attributes
    ----------
    data: xarray.Dataset
    settings: dict
        settings extracted from __init__ function
    geoax: cartopy.mpl.geoaxes.GeoAxes
        cartopy axes plotting georeferenced data
    fig: matplotlib.figure.Figure
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
        nlabel: Optional[int] = None,
        label: Optional[List[str]] = None,
        dpi: Number_T = 350,
        highlight: Optional[Union[str, List[str]]] = None,
        coastline: bool = False,
        extent: Optional[List[Number_T]] = None,
        section: Optional[Dataset] = None,
        style: str = "black",
        add_city_names: bool = False,
        plot_labels: bool = True,
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
        }
        if fig is None:
            self.fig = setup_plot(dpi, style=style)
        else:
            self.fig = fig
        # avoid in-place modification
        self.text_pos = TEXT_AXES_POS.copy()
        self.cbar_pos = CBAR_POS.copy()
        self._plot_ctx = dict()
        self._plot(**kwargs)

    def __call__(self, fpath: Optional[str] = None):
        if not fpath:
            # When the path is not specified, store the picture in home dir.
            fpath = os.path.join(str(Path.home()), "PyCINRAD")
        else:
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
                if nlabel:
                    clabel = np.linspace(n.vmin, n.vmax, nlabel).astype(str)
                else:
                    clabel = np.linspace(n.vmin, n.vmax, 10).astype(str)
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
        if not self.settings["extent"]:
            # 增加判断，城市名称绘制在选择区域内，否则自动绘制在data.lon和data.lat范围内
            self.settings["extent"] = [lon.min(), lon.max(), lat.min(), lat.max()]
        # When plot single radar, azimuthal equidistant projection is used.
        # The data which has code like 'Z9XXX' is considered as single radar.
        code = self.data.site_code
        if is_radial(self.data) and (code.startswith("Z") and code[1:].isnumeric()):
            proj = ccrs.AzimuthalEquidistant(
                central_longitude=self.data.site_longitude,
                central_latitude=self.data.site_latitude,
            )
        else:
            proj = ccrs.PlateCarree()
        self.geoax: GeoAxes = create_geoaxes(
            self.fig, proj, extent=self.settings["extent"]
        )
        if self.dtype in ["VEL", "SW"]:
            rf = self.data["RF"].values
        self._plot_ctx["var"] = var
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        self.geoax.pcolormesh(
            lon, lat, var, norm=pnorm, cmap=pcmap, transform=self.data_crs, **kwargs
        )
        if self.dtype in ["VEL", "SW"]:
            self.geoax.pcolormesh(
                lon,
                lat,
                rf,
                norm=norm_plot["RF"],
                cmap=cmap_plot["RF"],
                transform=self.data_crs,
                **kwargs
            )
        self._autoscale()
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

    def text(self):
        from cinrad.visualize.utils import plot_kw

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
        text(
            ax2,
            self.data.range,
            self.data.tangential_reso,
            self.data.scan_time,
            self.data.site_name,
            task,
            self.data.elevation,
        )
        ax2.text(0, INIT_TEXT_POS, prodname[self.dtype], **plot_kw)
        ax2.text(
            0,
            INIT_TEXT_POS - TEXT_SPACING * 8,
            "Max: {:.1f}{}".format(np.nanmax(var), unit[self.dtype]),
            **plot_kw
        )
        if self.dtype == "VEL":
            ax2.text(
                0,
                INIT_TEXT_POS - TEXT_SPACING * 9,
                "Min: {:.1f}{}".format(np.nanmin(var), unit[self.dtype]),
                **plot_kw
            )

    def _save(self, fpath: str):
        # Finalize texting here
        pnorm, cnorm, clabel = self._norm()
        pcmap, ccmap = self._cmap()
        if self.settings["plot_labels"]:
            self.text()
        cbar = setup_axes(self.fig, ccmap, cnorm, self.cbar_pos)
        if not isinstance(clabel, type(None)):
            change_cbar_text(
                cbar, np.linspace(cnorm.vmin, cnorm.vmax, len(clabel)), clabel
            )
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
                self.data.scan_time.strftime("%Y%m%d%H%M%S"),
                self.data.elevation,
                self.data.range,
                self.dtype.upper(),
                sec,
            )
        else:
            path_string = fpath
        save(path_string)

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

    def plot_cross_section(
        self,
        data: Dataset,
        ymax: Optional[int] = None,
        linecolor: Optional[str] = None,
        interpolate: bool = True,
    ):
        # May add check to ensure the data is slice data
        r"""Plot cross section data below the PPI plot."""
        if not linecolor:
            if self.settings["style"] == "black":
                linecolor = "white"
            elif self.settings["style"] == "white":
                linecolor = "black"
        self.settings["slice"] = data
        # The axes to plot c-section is below the main axes
        # the height of it is a quarter of the height of main axes
        # so the positions of the figure, the main axes, the colorbar axes
        # should be adjusted accordingly.
        # TODO: remove hardcode and calculate positions automatically
        self.fig.set_size_inches(10, 10)
        self.geoax.set_position([0, 0.2, 0.8, 0.8])
        ax2 = self.fig.add_axes([0, 0.01, 0.8, 0.17])
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
            ax2.pcolormesh(xcor, ycor, sl, cmap=cmap, norm=norm)
        if ymax:
            ax2.set_ylim(0, ymax)
        else:
            ax2.set_ylim(0, 15)
        ax2.set_title(
            "Start: {}N {}E".format(data.start_lat, data.start_lon)
            + " End: {}N {}E".format(data.end_lat, data.end_lon)
        )
        self.geoax.plot(
            [data.start_lon, data.end_lon],
            [data.start_lat, data.end_lat],
            marker="x",
            color=linecolor,
            transform=self.data_crs,
            zorder=5,
        )

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
        if not isinstance(self.geoax.projection, ccrs.PlateCarree):
            # Some workaround about the issue that cartopy version lower than 0.18 cannot
            # draw ticks on AzimuthalEquidistant plot
            from cartopy import __version__

            ver = tuple([i for i in __version__.split(".")])
            if ver < ("0", "18", "0"):
                warnings.warn(
                    "Cartopy older than 0.18 cannot draw ticks on AzimuthalEquidistant plot.",
                    RuntimeWarning,
                )
                return
        liner = self.geoax.gridlines(
            draw_labels=draw_labels,
            linewidth=linewidth,
            transform=self.data_crs,
            **kwargs
        )
        liner.xlabels_top = False
        liner.ylabels_right = False

    def _add_city_names(self):
        from cinrad.visualize.utils import MODULE_DIR, plot_kw

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
                **plot_kw,
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
