# https://github.com/crazyapril/mpkit/blob/master/gpf.py

import os
import sys
import ast

import matplotlib.colors as mclr
import numpy as np

from cinrad.constants import MODULE_DIR

_cmapdir_ = os.path.join(MODULE_DIR, "data", "colormap")

LAST_COLOR = 0
CONTROL_COLOR = 1
TRANSIT_COLOR = 2


def cmap(inp):
    """return cmap dict including plotplus control information ( levels and unit )"""
    c = Colormap(inp)
    return c.process()


def _cmap(inp):
    """return cmap dict without plotplus control information"""
    c = Colormap(inp)
    d = c.process()
    d.pop("levels", None)
    d.pop("unit", None)
    return d


def pure_cmap(inp):
    """return colormap instance"""
    return cmap(inp)["cmap"]


class ColormapDefinitionError(Exception):
    def __init__(self, description):
        self.dsc = description

    def __str__(self):
        return repr(self.dsc)


class Colormap:
    def __init__(self, inp):
        self.control = dict(type="auto", level="auto")
        if isinstance(inp, str):
            if inp.endswith(".gpf"):
                self.filepath = os.path.join(_cmapdir_, inp)
                self.process = self.gpf
            else:
                self.control.update(name=inp)
                self.process = self.cmap_linear
        elif isinstance(inp, dict):
            if "name" not in inp:
                raise ColormapDefinitionError('[Error 0] Need keyword "name" in dict.')
            self.control.update(inp)
            t = self.control["type"].lower()
            if t == "auto":
                self.process = self.cmap_linear
            elif t == "linear":
                self.process = self.cmap_linear
            elif t == "listed":
                self.process = self.cmap_listed
            else:
                raise ColormapDefinitionError("[Error 1] Unknown colormap type.")

    def cmap_listed(self):
        last_tval = -1e7
        tval_list = list()
        color_list = list()
        filepath = os.path.join(_cmapdir_, self.control["name"] + ".cmap")
        if not os.path.exists(filepath):
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])),
                self.control["name"] + ".cmap",
            )
        with open(filepath, "r") as f:
            transit_count = 0
            for line in f:
                if line[0] == "*":
                    lsplit = line[1:].split(":")
                    tname, tval = lsplit[0].lower(), lsplit[1][:-1]
                    if tname == "type":
                        if tval.lower() == "linear":
                            return self.cmap_linear()
                        elif tval.lower() != "listed":
                            raise ColormapDefinitionError(
                                "[Error 1] Unknown colormap type."
                            )
                    if tname not in self.control or self.control[tname] == "auto":
                        self.control[tname] = tval
                else:
                    tval, color = self.single_line_listed(line)
                    if tval < last_tval:
                        raise ColormapDefinitionError(
                            "[Error 5] tval should be arranged in order. Line:" + line
                        )
                    tval_list.append(tval)
                    if color == TRANSIT_COLOR:
                        color_list.append(0)
                        transit_count += 1
                    elif color != CONTROL_COLOR:
                        if transit_count > 0:
                            bcolor = color_list[-transit_count - 1]
                            for i in range(-transit_count, 0):
                                ratio = (transit_count + i + 1) / (transit_count + 1)
                                color_list[i] = tuple(
                                    (j - i) * ratio + i for i, j in zip(bcolor, color)
                                )
                            transit_count = 0
                        color_list.append(color)
                    last_tval = tval
        cmap = mclr.ListedColormap(color_list)
        unit = self.control.get("unit", None)
        over = self.control.get("over", None)
        extend = "neither"
        if over:
            cmap.set_over(self.get_color(over, "OVER"))
            extend = "max"
        under = self.control.get("under", None)
        if under:
            cmap.set_under(self.get_color(under, "UNDER"))
            extend = "min"
        if over and under:
            extend = "both"
        norm = mclr.BoundaryNorm(tval_list, cmap.N)
        return dict(cmap=cmap, levels=tval_list, norm=norm, unit=unit, extend=extend)

    def cmap_linear(self):
        last_color = None
        last_tval = -1e7
        datacache = list()
        filepath = os.path.join(_cmapdir_, self.control["name"] + ".cmap")
        if not os.path.exists(filepath):
            filepath = os.path.join(
                os.path.dirname(os.path.abspath(sys.argv[0])),
                self.control["name"] + ".cmap",
            )
        with open(filepath, "r") as f:
            for line in f:
                if line[0] == "*":
                    lsplit = line[1:].split(":")
                    tname, tval = lsplit[0].lower(), lsplit[1][:-1]
                    if tname == "type":
                        if tval.lower() == "listed":
                            return self.cmap_listed()
                        elif tval.lower() != "linear":
                            raise ColormapDefinitionError(
                                "[Error 1] Unknown colormap type."
                            )
                    if tname not in self.control or self.control[tname] == "auto":
                        self.control[tname] = tval
                else:
                    tval, color1, color2 = self.single_line_linear(line)
                    if tval < last_tval:
                        raise ColormapDefinitionError(
                            "[Error 5] tval should be arranged in order. Line:" + line
                        )
                    if color1 == LAST_COLOR:
                        color1 = last_color
                    if color2 == LAST_COLOR:
                        color2 = color1
                    datacache.append((tval, color1, color2))
                    last_tval = tval
                    last_color = color2
        tmin = datacache[0][0]
        tmax = datacache[-1][0]
        span = tmax - tmin
        colormap = {"red": [], "green": [], "blue": []}
        for tval, color1, color2 in datacache:
            trel = (tval - tmin) / span
            colormap["red"].append((trel, color1[0], color2[0]))
            colormap["green"].append((trel, color1[1], color2[1]))
            colormap["blue"].append((trel, color1[2], color2[2]))
        cmap = mclr.LinearSegmentedColormap("cmap", colormap)
        lvctrl = self.control["level"].lower()
        levels = self.get_levels(lvctrl, datacache)
        unit = self.control.get("unit", None)
        over = self.control.get("over", None)
        extend = "neither"
        if over:
            cmap.set_over(self.get_color(over, "OVER"))
            extend = "max"
        under = self.control.get("under", None)
        if under:
            cmap.set_under(self.get_color(under, "UNDER"))
            extend = "min"
        if over and under:
            extend = "both"
        return dict(cmap=cmap, levels=levels, unit=unit, extend=extend)

    def single_line_linear(self, line):
        if line[-1] == "\n":
            line = line[:-1]
        lsplit = line.split(" ")
        if len(lsplit) != 3:
            raise ColormapDefinitionError(
                "[Error 2] Only 2 spaces are allowed in a line. Line:" + line
            )
        try:
            tval = float(lsplit[0])
        except (SyntaxError, ValueError, NameError):
            raise ColormapDefinitionError("[Error 3] Invalid value. Line:" + line)
        if lsplit[1] == "BEGIN":
            return tval, (0.0, 0.0, 0.0), self.get_color(lsplit[2], line)
        elif lsplit[2] == "END":
            return tval, self.get_color(lsplit[1], line), (0.0, 0.0, 0.0)
        else:
            return (
                tval,
                self.get_color(lsplit[1], line),
                self.get_color(lsplit[2], line),
            )

    def single_line_listed(self, line):
        if line[-1] == "\n":
            line = line[:-1]
        lsplit = line.split(" ")
        if len(lsplit) != 2:
            raise ColormapDefinitionError(
                "[Error 2] Only 1 space are allowed in a line. Line:" + line
            )
        try:
            tval = float(lsplit[0])
        except (SyntaxError, ValueError, NameError):
            raise ColormapDefinitionError("[Error 3] Invalid value. Line:" + line)
        if lsplit[1] == "END":
            return tval, CONTROL_COLOR
        elif lsplit[1] == "~":
            return tval, TRANSIT_COLOR
        else:
            return tval, self.get_color(lsplit[1], line)

    def get_levels(self, lvctrl, datacache):
        tmin = datacache[0][0]
        tmax = datacache[-1][0]
        if lvctrl == "file":
            levels = list()
            for r in datacache:
                levels.append(r[0])
        elif lvctrl.startswith("s"):
            lvctrl = int(lvctrl[1:])
            levels = list()
            for i in range(len(datacache) - 1):
                levels.extend(np.linspace(datacache[i][0], datacache[i + 1][0], lvctrl))
        elif lvctrl.startswith("["):
            levels = ast.literal_eval(lvctrl)
        else:
            if lvctrl == "auto":
                lvctrl = 1
            elif lvctrl.startswith("c"):
                lvctrl = lvctrl[1:]
            levels = np.arange(tmin, tmax + float(lvctrl), float(lvctrl))
        return levels

    def get_color(self, l, line):
        if l == "~":
            return LAST_COLOR
        b = l.split("/")
        if len(b) != 3:
            raise ColormapDefinitionError(
                "[Error 4] Invalid color format. Line:" + line
            )
        for i in range(len(b)):
            v = b[i]
            try:
                v = float(v)
            except (SyntaxError, ValueError, NameError):
                raise ColormapDefinitionError("[Error 3] Invalid value. Line:" + line)
            v /= 255.0
            if v < 0 or v > 1:
                raise ColormapDefinitionError(
                    "[Error 6] Value out of range (0~255). Line:" + line
                )
            b[i] = v
        return tuple(b)

    def gpf(self):
        cmap = {"red": [], "green": [], "blue": []}
        with open(self.filepath, "r") as f:
            lastred = (0.0, 0.0, 0.0)
            lastgreen = lastred
            lastblue = lastred
            line = f.readline()
            while line:
                if line[0] != "#":
                    data = [ast.literal_eval(numbyte) for numbyte in line.split()]
                    red = (data[0], lastred[2], data[1])
                    green = (data[0], lastgreen[2], data[2])
                    blue = (data[0], lastblue[2], data[3])
                    cmap["red"].append(red)
                    cmap["green"].append(green)
                    cmap["blue"].append(blue)
                    lastred = red
                    lastgreen = green
                    lastblue = blue
                line = f.readline()
        return dict(cmap=mclr.LinearSegmentedColormap("gpf", cmap))


if __name__ == "__main__":
    from matplotlib.colorbar import ColorbarBase
    import matplotlib.pyplot as plt

    while True:
        cmapname = input("Colormap name>")
        if not cmapname:
            break
        plt.figure(1, figsize=(11, 2))
        ax = plt.gca()
        cmapdict = _cmap(cmapname)
        cmapdict.update(orientation="horizontal")
        ColorbarBase(ax, **cmapdict)
        plt.show()
        # plt.clf()
