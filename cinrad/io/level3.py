# -*- coding: utf-8 -*-
# Author: Puyuan Du

from collections import OrderedDict, defaultdict
from typing import Union, Any
import datetime
import os
import glob
from io import BytesIO

import numpy as np

from cinrad.projection import get_coordinate
from cinrad.constants import deg2rad
from cinrad._typing import Boardcast_T
from cinrad.io.base import RadarBase, prepare_file
from cinrad.io._dtype import *
from cinrad.datastruct import Radial, Grid
from cinrad.error import RadarDecodeError


def xy2polar(x: Boardcast_T, y: Boardcast_T) -> tuple:
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y) * 180 / np.pi


class PUP(RadarBase):
    r"""
    Class handling PUP data (Nexrad Level III data)
    """

    def __init__(self, file: Any):
        from metpy.io.nexrad import Level3File

        f = Level3File(file)
        self.dtype = self._det_product_type(f.prod_desc.prod_code)
        self.radial_flag = self._is_radial(f.prod_desc.prod_code)
        data_block = f.sym_block[0][0]
        data = np.ma.array(data_block["data"])
        data[data == 0] = np.ma.masked
        self.data = np.ma.masked_invalid(f.map_data(data))
        self.max_range = f.max_range
        if self.radial_flag:
            self.az = (
                np.array(data_block["start_az"] + [data_block["end_az"][-1]]) * deg2rad
            )
            self.rng = np.linspace(0, f.max_range, data.shape[-1] + 1)
            self.reso = self.max_range / data.shape[1]
        else:
            # TODO: Support grid type data
            raise NotImplementedError("Grid-type data is not supported")
            xdim, ydim = data.shape
            x = (
                np.linspace(xdim * f.ij_to_km * -1, xdim * f.ij_to_km, xdim) / 111
                + f.lon
            )
            y = (
                np.linspace(ydim * f.ij_to_km, ydim * f.ij_to_km * -1, ydim) / 111
                + f.lat
            )
            self.lon, self.lat = np.meshgrid(x, y)
            self.reso = f.ij_to_km
        self.stationlat = f.lat
        self.stationlon = f.lon
        self.el = f.metadata["el_angle"]
        self.scantime = f.metadata["vol_time"]
        # Because metpy interface doesn't provide station codes,
        # it's necessary to reopen it and read the code.
        o = open(file, "rb")
        o.seek(12)
        code = np.frombuffer(o.read(2), ">i2")[0]
        if code in range(0, 100):
            cds = "0{}".format(code)
        else:
            cds = str(code)
        self.code = "Z9" + cds
        o.close()
        self._update_radar_info()

    def get_data(self) -> Union[Grid, Radial]:
        if self.radial_flag:
            lon, lat = self.projection()
            return Radial(
                self.data,
                self.max_range,
                self.el,
                self.reso,
                self.code,
                self.name,
                self.scantime,
                self.dtype,
                self.stationlon,
                self.stationlat,
                lon,
                lat,
            )
        else:
            return Grid(
                self.data,
                self.max_range,
                self.reso,
                self.code,
                self.name,
                self.scantime,
                self.dtype,
                self.stationlon,
                self.stationlat,
                self.lon,
                self.lat,
            )

    @staticmethod
    def _is_radial(code: int) -> bool:
        return code in range(16, 31)

    def projection(self) -> tuple:
        return get_coordinate(
            self.rng, self.az, self.el, self.stationlon, self.stationlat, h_offset=False
        )

    @staticmethod
    def _det_product_type(spec: int) -> str:
        if spec in range(16, 22):
            return "REF"
        elif spec in range(22, 28):
            return "VEL"
        elif spec in range(28, 31):
            return "SW"
        elif spec == 37:
            return "CR"
        else:
            raise RadarDecodeError("Unsupported product type {}".format(spec))


class SWAN(object):
    dtype_conv = {0: "B", 1: "b", 2: "u2", 3: "i2", 4: "u2"}
    size_conv = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}

    def __init__(self, file: Any):
        f = prepare_file(file)
        header = np.frombuffer(f.read(1024), SWAN_dtype)
        xdim, ydim, zdim = (
            header["x_grid_num"][0],
            header["y_grid_num"][0],
            header["z_grid_num"][0],
        )
        dtype = header["m_data_type"][0]
        data_size = int(xdim) * int(ydim) * int(zdim) * self.size_conv[dtype]
        bittype = self.dtype_conv[dtype]
        data_body = np.frombuffer(f.read(data_size), bittype).astype(int)
        # Convert data to i4 to avoid overflow in later calculation
        if zdim == 1:
            # 2D data
            out = data_body.reshape(xdim, ydim)
        else:
            # 3D data
            out = data_body.reshape(zdim, ydim, xdim)
        self.data_time = datetime.datetime(
            header["year"],
            header["month"],
            header["day"],
            header["hour"],
            header["minute"],
        )
        # TODO: Recognize correct product name
        self.product_name = b"".join(header["data_name"]).decode().replace("\x00", "")
        start_lon = header["start_lon"][0]
        start_lat = header["start_lat"][0]
        center_lon = header["center_lon"][0]
        center_lat = header["center_lat"][0]
        end_lon = center_lon * 2 - start_lon
        end_lat = center_lat * 2 - start_lat
        # x_reso = header['x_reso'][0]
        # y_reso = header['y_reso'][0]
        self.lon = np.linspace(start_lon, end_lon, xdim)  # For shape compatibility
        self.lat = np.linspace(start_lat, end_lat, ydim)
        if self.product_name in ["CR", "3DREF"]:
            self.data = (np.ma.masked_equal(out, 0) - 66) / 2
        else:
            # Leave data unchanged because the scale and offset are unclear
            self.data = np.ma.masked_equal(out, 0)

    def get_data(self, level=0) -> Grid:
        x, y = np.meshgrid(self.lon, self.lat)
        dtype = self.product_name
        if self.data.ndim == 2:
            ret = self.data
        else:
            ret = self.data[level]
            if self.product_name == "3DREF":
                dtype = "CR"
        grid = Grid(
            ret, np.nan, np.nan, "SWAN", "SWAN", self.data_time, dtype, 0, 0, x, y
        )
        return grid


class StormTrackInfo(object):
    def __init__(self, filepath: str):
        from metpy.io.nexrad import Level3File

        self.handler = Level3File(filepath)
        self.info = self.get_all_sti()
        self.storm_list = self.get_all_id()

    def get_all_sti(self) -> OrderedDict:
        f = self.handler
        if not hasattr(f, "sym_block"):
            return OrderedDict()
        else:
            data_block = f.sym_block[0]
            sti_data = OrderedDict()
            data_dict = [i for i in data_block if isinstance(i, defaultdict)]
            for i in data_dict:
                if i["type"] == "Storm ID":
                    sti_data[i["id"]] = defaultdict()
                    sti_data[i["id"]]["current storm position"] = tuple(
                        [i["x"], i["y"]]
                    )
                else:
                    stid = list(sti_data.keys())[-1]
                    if "markers" in i.keys():
                        pos = i["markers"]
                        if isinstance(pos, dict):
                            pos = [pos]
                        name = list(pos[0].keys())[0]
                        sti_data[stid][name] = list()
                        sti_data[stid][name] += i.get("track")
                    elif "STI Circle" in i.keys():
                        circle_dict = i["STI Circle"]
                        sti_data[stid]["radius"] = circle_dict["radius"]
                        sti_data[stid]["current storm position"] = tuple(
                            [circle_dict["x"], circle_dict["y"]]
                        )
        return sti_data

    def get_all_id(self) -> list:
        return list(self.info.keys())

    def current(self, storm_id: str) -> tuple:
        curpos = self.info[storm_id]["current storm position"]
        dist, az = xy2polar(*curpos)
        lonlat = get_coordinate(
            dist, az * deg2rad, 0, self.handler.lon, self.handler.lat, h_offset=False
        )
        return lonlat

    def track(self, storm_id: str, tracktype: str) -> tuple:
        if tracktype == "forecast":
            key = "forecast storm position"
        elif tracktype == "past":
            key = "past storm position"
        else:
            raise KeyError("Key {} does not exist".format(key))
        if key not in self.info[storm_id].keys():
            return None
        forpos = self.info[storm_id][key]
        if forpos == None:
            return
        x_pos = np.array([i[0] for i in forpos])
        y_pos = np.array([i[1] for i in forpos])
        pol_pos = xy2polar(x_pos, y_pos)
        lon = list()
        lat = list()
        for dis, azi in zip(pol_pos[0], pol_pos[1]):
            pos_tup = get_coordinate(
                dis,
                azi * deg2rad,
                0,
                self.handler.lon,
                self.handler.lat,
                h_offset=False,
            )
            lon.append(pos_tup[0])
            lat.append(pos_tup[1])
        return np.array(lon), np.array(lat)


class HailIndex(object):
    def __init__(self, filepath: str):
        from metpy.io.nexrad import Level3File

        self.handler = Level3File(filepath)
        self.info = self.get_all_hi()
        self.storm_list = self.get_all_id()

    def get_all_hi(self) -> OrderedDict:
        f = self.handler
        if not hasattr(f, "sym_block"):
            return OrderedDict()
        else:
            data_block = f.sym_block[0]
            sti_data = OrderedDict()
            data_dict = [i for i in data_block if isinstance(i, defaultdict)]
            storm_id = [i for i in data_dict if i["type"] == "Storm ID"]
            info = [i for i in data_dict if i["type"] == "HDA"]
            for sid, inf in zip(storm_id, info):
                stid = sid["id"]
                sti_data[stid] = defaultdict()
                sti_data[stid]["current storm position"] = tuple([sid["x"], sid["y"]])
                sti_data[stid]["POH"] = inf["POH"]
                sti_data[stid]["POSH"] = inf["POSH"]
                sti_data[stid]["MESH"] = inf["Max Size"]
            return sti_data

    def get_all_id(self) -> list:
        return list(self.info.keys())

    def get_hail_param(self, storm_id: str) -> dict:
        out = dict(self.info[storm_id])
        xy = out.get("current storm position")
        dist, az = xy2polar(*xy)
        lonlat = get_coordinate(
            dist, az * deg2rad, 0, self.handler.lon, self.handler.lat, h_offset=False
        )
        out["position"] = tuple(lonlat)
        return out


class _ProductParams(object):
    def __init__(self, ptype: int, param_bytes: bytes):
        self.buf = BytesIO(param_bytes)
        self.params = dict()
        map_func = {1: self._ppi, 2: self._rhi}
        map_func[ptype]()
        self.buf.close()

    def _ppi(self):
        elev = np.frombuffer(self.buf.read(4), "f4")[0]
        self.params["elevation"] = elev

    def _rhi(self):
        azi = np.frombuffer(self.buf.read(4), "f4")[0]
        top = np.frombuffer(self.buf.read(4), "f4")[0]
        bot = np.frombuffer(self.buf.read(4), "f4")[0]
        self.params["azimuth"] = azi
        self.params["top"] = top
        self.params["bottom"] = bot


def get_product_param(ptype: int, param_bytes: bytes) -> dict:
    return _ProductParams(ptype, param_bytes).params


class StandardPUP(RadarBase):

    # fmt: off
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 32:'Zc', 33:'Vc', 34:'Wc', 35:'ZDRc', 71:'RR', 72:'HGT',
                  73:'VIL', 74:'SHR', 75:'RAIN', 76:'RMS', 77:'CTR'}
    # fmt: on
    def __init__(self, file):
        self.f = prepare_file(file)
        self._parse()
        self._update_radar_info()
        self.stationlat = self.geo["lat"][0]
        self.stationlon = self.geo["lon"][0]
        self.radarheight = self.geo["height"][0]
        if self.name == "None":
            self.name = self.code
        del self.geo
        self.f.close()

    def _parse(self):
        header = np.frombuffer(self.f.read(32), SDD_header)
        if header["magic_number"] != 0x4D545352:
            raise RadarDecodeError("Invalid standard data")
        site_config = np.frombuffer(self.f.read(128), SDD_site)
        self.code = b"".join(site_config["site_code"][0]).decode().replace("\x00", "")
        self.geo = geo = dict()
        geo["lat"] = site_config["Latitude"]
        geo["lon"] = site_config["Longitude"]
        geo["height"] = site_config["ground_height"]
        task = np.frombuffer(self.f.read(256), SDD_task)
        self.task_name = b"".join(task["task_name"][0]).decode().replace("\x00", "")
        cut_num = task["cut_number"][0]
        scan_config = np.frombuffer(self.f.read(256 * cut_num), SDD_cut)
        ph = np.frombuffer(self.f.read(128), SDD_pheader)
        ptype = ph["product_type"][0]
        self.scantime = datetime.datetime(*time.gmtime(ph["scan_start_time"])[:6])
        self.dtype = self.dtype_corr[ph["dtype_1"][0]]
        params = get_product_param(ptype, self.f.read(64))

        radial_header = np.frombuffer(self.f.read(64), L3_radial)
        bin_length = radial_header["bin_length"][0]
        scale = radial_header["scale"][0]
        offset = radial_header["offset"][0]
        self.reso = radial_header["reso"][0] / 1000
        self.start_range = radial_header["start_range"][0] / 1000
        self.end_range = radial_header["max_range"][0] / 1000
        data = list()
        azi = list()
        while True:
            buf = self.f.read(32)
            if not buf:
                break
            data_block = np.frombuffer(buf, L3_rblock)
            start_a = data_block["start_az"][0]
            nbins = data_block["nbins"][0]
            raw = np.frombuffer(
                self.f.read(bin_length * nbins), "u{}".format(bin_length)
            )
            data.append(raw)
            azi.append(start_a)
        raw = np.vstack(data).astype(int)
        raw = np.ma.masked_less_equal(raw, 5)
        self.data = (raw - offset) / scale
        self.el = params["elevation"]
        self.azi = np.deg2rad(azi)

    def get_data(self):
        dist = np.arange(self.start_range, self.end_range, self.reso)
        lon, lat = get_coordinate(
            dist, self.azi, self.el, self.stationlon, self.stationlat
        )
        hgt = height(dist, self.el, self.radarheight)
        radial = Radial(
            self.data,
            self.end_range,
            self.el,
            self.reso,
            self.code,
            self.name,
            self.scantime,
            self.dtype,
            self.stationlon,
            self.stationlat,
            lon,
            lat,
            height,
            task=self.task_name,
        )
        return radial
