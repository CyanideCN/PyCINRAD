# -*- coding: utf-8 -*-
# Author: Puyuan Du

import os
from datetime import datetime
from typing import Union, Optional, Any, List
from functools import lru_cache

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.lines import Line2D
import matplotlib.colors as cmx
from matplotlib.font_manager import FontProperties
import matplotlib.cm as mcm
from cartopy.io import shapereader
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.feature import Feature
import shapefile
import shapely.geometry as sgeom
from vanadis.colormap import Colormap
from cinrad_data import get_font_path, get_shp_list, get_shp_file

from cinrad.visualize.gpf import _cmap
from cinrad.utils import MODULE_DIR
from cinrad._typing import Array_T, Number_T
from cinrad.error import RadarPlotError
from cinrad.visualize.layout import (
    FIG_SIZE,
    GEOAXES_POS,
    INIT_TEXT_POS,
    TEXT_SPACING,
)

__all__ = [
    "add_shp",
    "save",
    "setup_axes",
    "setup_plot",
    "text",
    "change_cbar_text",
    "draw_highlight_area",
    "create_geoaxes",
    "norm_plot",
    "norm_cbar",
    "cmap_plot",
    "cmap_cbar",
    "sec_plot",
    "prodname",
    "unit",
    "cbar_text",
    "is_inline",
    "plot_kw",
]

CMAP_DIR = os.path.join(MODULE_DIR, "data", "colormap")


def _get_uniform_cmap(cmap: Any) -> Any:
    new_cm = Colormap(cmap=cmap.reversed()).set_uniform()
    return new_cm.as_mpl_cmap()


r_cmap = _cmap("REF")["cmap"]
r_cmap_smooth = _cmap("REF_s")["cmap"]
v_cmap = _cmap("VEL")["cmap"]
v_cbar = _cmap("VEL_reverse")["cmap"]
v_cmap_smooth = _cmap("VEL_s")["cmap"]
zdr_cmap = _cmap("ZDR")["cmap"]
zdr_cbar = _get_uniform_cmap(zdr_cmap)
zdr_cmap_smooth = _cmap("ZDR_s")["cmap"]
kdp_cmap = _cmap("KDP")["cmap"]
kdp_cbar = _get_uniform_cmap(kdp_cmap)
kdp_cmap_smooth = _cmap("KDP_s")["cmap"]
cc_cmap = _cmap("CC")["cmap"]
cc_cbar = _get_uniform_cmap(cc_cmap)
cc_cmap_smooth = _cmap("CC_s")["cmap"]
et_cmap = _cmap("ET")["cmap"]
et_cbar = _get_uniform_cmap(et_cmap)
vil_cmap = _cmap("VIL")["cmap"]
vil_cbar = _get_uniform_cmap(vil_cmap)
rf_cmap = cmx.ListedColormap("#660066", "#FFFFFF")
ohp_cmap = _cmap("OHP")["cmap"]
ohp_cbar = _get_uniform_cmap(ohp_cmap)
hcl_cmap = _cmap("HCL")["cmap"]
hcl_cbar = _get_uniform_cmap(hcl_cmap.reversed())

norm1 = cmx.Normalize(0, 75)  # reflectivity / vertially integrated liquid
norm2 = cmx.Normalize(-35, 28)  # velocity
norm3 = cmx.Normalize(-1, 0)  # RF
norm4 = cmx.Normalize(0, 1)  # colorbar
norm5 = cmx.Normalize(0, 21)  # echo top
norm6 = cmx.Normalize(-4, 5)  # differential reflectivity
norm7 = cmx.Normalize(260, 360)  # differential phase
norm8 = cmx.Normalize(0, 0.99)  # correlation coefficient
norm9 = cmx.Normalize(-0.8, 21)  # specific differential phase
norm10 = cmx.Normalize(0.1, 6)  # vertically integrated liquid density
norm11 = cmx.Normalize(0, 204)  # One-hour precipitation
norm12 = cmx.Normalize(1, 11)
norm13 = cmx.Normalize(0, 10)  # standard rose HCl

# fmt: off
norm_plot = {"REF":norm1, "VEL":norm2, "CR":norm1, "ET":norm5, "VIL":norm1, "RF":norm3,
             "ZDR":norm6, "PHI":norm7, "RHO":norm8, "TREF":norm1, "KDP":norm9, "VILD":norm10,
             "OHP":norm11, "cHCL":norm12, "HCL":norm13} # Normalize object used to plot
norm_cbar = {"REF":norm1, "VEL":norm4, "CR":norm1, "ET":norm4, "VIL":norm4,
             "ZDR":norm4, "PHI":norm4, "RHO":norm4, "TREF":norm1, "KDP":norm4,
             "VILD":norm4, "OHP":norm4, "cHCL":norm4, "HCL":norm4} # Normalize object used for colorbar
cmap_plot = {"REF":r_cmap, "VEL":v_cmap, "CR":r_cmap, "ET":et_cmap, "VIL":vil_cmap, "RF":rf_cmap,
             "ZDR":zdr_cmap, "PHI":kdp_cmap, "RHO":cc_cmap, "TREF":r_cmap, "KDP":kdp_cmap,
             "VILD":vil_cmap, "OHP":ohp_cmap, "cHCL":mcm.tab10, "HCL":hcl_cmap}
cmap_cbar = {"REF":r_cmap, "VEL":v_cbar, "CR":r_cmap, "ET":et_cbar, "VIL":vil_cbar,
             "ZDR":zdr_cbar, "PHI":kdp_cbar, "RHO":cc_cbar, "TREF":r_cmap, "KDP":kdp_cbar,
             "VILD":vil_cbar, "OHP":ohp_cbar, "cHCL":mcm.tab10, "HCL":hcl_cbar}
sec_plot = {"REF":r_cmap_smooth, "VEL":v_cmap_smooth, "ZDR":zdr_cmap_smooth, "PHI":kdp_cmap_smooth, "RHO":cc_cmap_smooth,
            "KDP":kdp_cmap_smooth, "cHCL":mcm.tab10, "HCL":hcl_cbar}
prodname = {"REF":"Base Reflectivity", "VEL":"Base Velocity", "CR":"Composite Ref.",
            "ET":"Echo Tops", "VIL":"V Integrated Liquid", "ZDR":"Differential Ref.",
            "PHI":"Differential Phase", "RHO":"Correlation Coe.", "TREF":"Total Reflectivity",
            "KDP":"Spec. Diff. Phase", "VILD":"VIL Density", "OHP":"One-Hour Precip.",
            "cHCL":"Hydrometeor Class","HCL":"Hydrometeor Class", "VELSZ":"Velocity SZ Recovery"}
unit = {"REF":"dBZ", "VEL":"m/s", "CR":"dBZ", "ET":"km", "VIL":"kg/m**2", "ZDR":"dB", "PHI":"deg",
        "RHO":"", "TREF":"dBZ", "KDP":"deg/km", "VILD":"g/m**3", "OHP":"mm", "cHCL":"", "HCL":""}
cbar_text = {"REF":None, "VEL":["RF", "", "27", "20", "15", "10", "5", "1", "0",
                                "-1", "-5", "-10", "-15", "-20", "-27", "-35"],
             "CR":None, "ET":["", "21", "20", "18", "17", "15", "14", "12",
                              "11", "9", "8", "6", "5", "3", "2", "0"],
             "VIL":["", "70", "65", "60", "55", "50", "45", "40", "35", "30",
                    "25", "20", "15", "10", "5", "0"],
             "ZDR":["", "5", "4", "3.5", "3", "2.5", "2", "1.5", "1", "0.8", "0.5",
                    "0.2", "0", "-1", "-2", "-3", "-4"],
             "PHI":np.linspace(360, 260, 17).astype(str),
             "RHO":["", "0.99", "0.98", "0.97", "0.96", "0.95", "0.94", "0.92", "0.9",
                    "0.85", "0.8", "0.7", "0.6", "0.5", "0.3", "0.1", "0"],
             "TREF":None, "KDP":["", "20", "7", "3.1", "2.4", "1.7", "1.1", "0.75", "0.5",
                                 "0.33", "0.22", "0.15", "0.1", "-0.1", "-0.2", "-0.4", "-0.8"],
             "VILD":["", "6", "5", "4", "3.5", "3", "2.5", "2.1", "1.8", "1.5", "1.2",
                     "0.9", "0.7", "0.5", "0.3", "0.1"],
             "OHP":["", "203.2", "152.4", "101.6", "76.2", "63.5", "50.8", "44.45", "38.1", "31.75",
                    "25.4", "19.05", "12.7", "6.35", "2.54", "0"],
             "cHCL":["Drizzle", "Rain", "Ice Crystals", "Dry Snow", "Wet Snow", "Vertical Ice",
                     "Low-Dens Graupel", "High-Dens Graupel", "Hail", "Big Drops", ""],
             "HCL":["Rain", "Heavy Rain", "Hail", "Big Drops", "Clear-Air Echo", "Ground Clutter",
                    "Dry snow", "Wet snow", "Ice Crystals", "Graupel", "Unknown", ""]}
# fmt: on

# Add entry for VELSZ
for dic in zip([norm_plot, norm_cbar, cmap_plot, cmap_cbar, sec_plot, unit, cbar_text]):
    _d = dic[0]
    _d["VELSZ"] = _d["VEL"]

font = FontProperties(fname=get_font_path())
plot_kw = {"fontproperties": font, "fontsize": 12}


def set_font(font_path: str):
    glb = globals()
    font = FontProperties(fname=font_path)
    glb["plot_kw"].update({"fontproperties": font})


class ShpReader(shapereader.BasicReader):
    r"""Customized reader to deal with encoding issue"""

    def __init__(self, filename: str, encoding: str = "gbk"):
        # Validate the filename/shapefile
        self._reader = reader = shapefile.Reader(filename, encoding=encoding)
        if reader.shp is None or reader.shx is None or reader.dbf is None:
            raise ValueError("Incomplete shapefile definition " "in '%s'." % filename)
        try:
            shapeType = reader.shapeType
            self._geometry_factory = shapereader.GEOMETRY_FACTORIES.get(shapeType)
            if self._geometry_factory is None:
                raise ValueError("Unsupported shape type: %s" % shapeType)
        except AttributeError:
            pass
        self._fields = self._reader.fields


from cartopy import __version__

if __version__ >= "0.23.0":
    ShpReader = shapereader.BasicReader


def setup_plot(dpi: Number_T, figsize: tuple = FIG_SIZE, style: str = "black") -> Any:
    if style == "transparent":
        figsize = (10, 10)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.axis("off")
    if style == "black":
        plt.style.use("dark_background")
    return fig


def setup_axes(fig: Any, cmap: Any, norm: Any, position: List[Number_T]) -> tuple:
    ax = fig.add_axes(position)
    cbar = ColorbarBase(
        ax, cmap=cmap, norm=norm, orientation="vertical", drawedges=False
    )
    cbar.ax.tick_params(axis="both", which="both", length=0, labelsize=10)
    cbar.outline.set_visible(False)
    return cbar


def text(
    ax: Any,
    drange: Number_T,
    reso: float,
    scantime: str,
    name: str,
    task: str,
    elev: float,
):
    from cinrad.visualize.utils import plot_kw

    ax.text(
        0, INIT_TEXT_POS - TEXT_SPACING, "Range: {:.0f}km".format(drange), **plot_kw
    )
    if reso < 0.1:
        # Change the unit from km to m for better formatting
        ax.text(
            0,
            INIT_TEXT_POS - TEXT_SPACING * 2,
            "Resolution: {:.0f}m".format(reso * 1000),
            **plot_kw
        )
    else:
        ax.text(
            0,
            INIT_TEXT_POS - TEXT_SPACING * 2,
            "Resolution: {:.2f}km".format(reso),
            **plot_kw
        )
    ax.text(
        0,
        INIT_TEXT_POS - TEXT_SPACING * 3,
        "Date: {}".format(
            datetime.strptime(scantime, "%Y-%m-%d %H:%M:%S").strftime("%Y.%m.%d")
        ),
        **plot_kw
    )
    ax.text(
        0,
        INIT_TEXT_POS - TEXT_SPACING * 4,
        "Time: {}".format(
            datetime.strptime(scantime, "%Y-%m-%d %H:%M:%S").strftime("%H:%M")
        ),
        **plot_kw
    )
    if name is None:
        name = "Unknown"
    ax.text(0, INIT_TEXT_POS - TEXT_SPACING * 5, "RDA: " + name, **plot_kw)
    ax.text(0, INIT_TEXT_POS - TEXT_SPACING * 6, "Task: {}".format(task), **plot_kw)
    ax.text(
        0, INIT_TEXT_POS - TEXT_SPACING * 7, "Elev: {:.2f}deg".format(elev), **plot_kw
    )


def save(fpath: str, style: str = "black", **kwargs):
    if style == "transparent":
        plt.savefig(fpath, transparent=True, pad_inches=0, **kwargs)
    else:
        plt.savefig(fpath, pad_inches=0, **kwargs)
    plt.close("all")


@lru_cache(maxsize=2)
def get_shp() -> list:
    flist = get_shp_list()
    shps = [list(ShpReader(i, encoding="gbk").geometries()) for i in flist]
    return shps


class _ShapelyFeature(Feature):
    r"""Copied from cartopy.feature.ShapelyFeature"""

    def __init__(self, geometries, crs, **kwargs):
        super().__init__(crs, **kwargs)
        if isinstance(geometries, sgeom.base.BaseGeometry):
            geometries = [geometries]
        self._geoms = tuple(geometries)

    def geometries(self):
        return iter(self._geoms)


def add_shp(
    ax: Any,
    proj: ccrs.Projection,
    coastline: bool = False,
    style: str = "black",
    extent: Optional[Array_T] = None,
):
    if style == "transparent":
        return
    shp_crs = ccrs.PlateCarree()
    shps = get_shp()
    if style == "black":
        line_colors = ["grey", "lightgrey", "white"]
    elif style == "white":
        line_colors = ["lightgrey", "grey", "black"]
    ax.add_feature(
        _ShapelyFeature(
            shps[0],
            shp_crs,
            edgecolor=line_colors[0],
            facecolor="None",
            zorder=3,
            linewidth=0.5,
        )
    )
    ax.add_feature(
        _ShapelyFeature(
            shps[1],
            shp_crs,
            edgecolor=line_colors[1],
            facecolor="None",
            zorder=3,
            linewidth=0.7,
        )
    )
    ax.add_feature(
        _ShapelyFeature(
            shps[2],
            shp_crs,
            edgecolor=line_colors[2],
            facecolor="None",
            zorder=3,
            linewidth=1,
        )
    )
    if coastline:
        ax.coastlines(resolution="10m", color=line_colors[2], zorder=3, linewidth=1)


def change_cbar_text(cbar: ColorbarBase, tick: List[Number_T], text: List[str]):
    cbar.set_ticks(tick)
    cbar.set_ticklabels(text, **plot_kw)


def highlight_area(
    area: Union[Array_T, str], linecolor: str = "red", **kwargs
) -> List[Line2D]:
    r"""Return list of Line2D object for given area name"""
    fpath = get_shp_file("City")
    shp = shapefile.Reader(fpath, encoding="gbk")
    rec = shp.shapeRecords()
    lines = list()
    if isinstance(area, str):
        area = [area]
    for i in area:
        if not isinstance(i, str):
            raise RadarPlotError("Area name should be str")
        name = np.array([i.record[2] for i in rec])
        target = np.array(rec)[(name == i).nonzero()[0]]
        for j in target:
            pts = j.shape.points
            x = [i[0] for i in pts]
            y = [i[1] for i in pts]
            lines.append(Line2D(x, y, color=linecolor))
    return lines


def draw_highlight_area(area: Union[Array_T, str]):
    lines = highlight_area(area)
    ax_ = plt.gca()
    for l in lines:
        pat = ax_.add_artist(l)
        pat.set_zorder(4)


def create_geoaxes(
    fig: Any, proj: ccrs.Projection, extent: List[Number_T], style: str = "black"
) -> GeoAxes:
    from cartopy import __version__

    if style == "transparent":
        ax = fig.add_axes([0, 0, 1, 1], projection=proj)
        ax.set_aspect("equal")
    else:
        ax = fig.add_axes(GEOAXES_POS, projection=proj)
    if __version__ < "0.18":
        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
    else:
        ax.patch.set_visible(False)
        ax.spines["geo"].set_visible(False)
    x_min, x_max, y_min, y_max = extent[0], extent[1], extent[2], extent[3]
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())
    return ax


def is_inline() -> bool:
    return "inline" in mpl.get_backend()
