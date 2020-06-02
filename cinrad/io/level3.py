# -*- coding: utf-8 -*-
# Author: Puyuan Du

from collections import OrderedDict, defaultdict
from typing import Union, Any
import datetime
from io import BytesIO

import numpy as np
from xarray import Dataset, DataArray

from cinrad.projection import get_coordinate
from cinrad.constants import deg2rad
from cinrad._typing import Boardcast_T
from cinrad.io.base import RadarBase, prepare_file, _get_radar_info
from cinrad.io._dtype import *
from cinrad.error import RadarDecodeError


def xy2polar(x: Boardcast_T, y: Boardcast_T) -> tuple:
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y) * 180 / np.pi


# As metpy use a different table to map the data
# The color table constructed from PUP software is used to replace metpy
# fmt: off
velocity_map = {0: np.ma.masked, 1: -27.5, 2: -20.5, 3: -15.5, 4: -10.5, 5: -5.5,
                6: -1.5, 7: -0.5, 8: 0.5, 9: 4.5, 10: 9.5, 11: 14.5, 12: 19.5,
                13: 26.5, 14: 27.5, 15: 30}
# fmt: on
velocity_mapper = np.vectorize(velocity_map.__getitem__)


class PUP(RadarBase):
    r"""
    Class handling PUP data (Nexrad Level III data)
    """

    def __init__(self, file: Any):
        from metpy.io.nexrad import Level3File

        f = Level3File(file)
        # Because metpy interface doesn't provide station codes,
        # it's necessary to reopen it and read the code.
        with open(file, "rb") as buf:
            buf.seek(12)
            code = np.frombuffer(buf.read(2), ">i2")[0]
            cds = str(code).zfill(3)
            self.code = "Z9" + cds
        self._update_radar_info()
        product_code = f.prod_desc.prod_code
        self.dtype = self._det_product_type(product_code)
        self.radial_flag = self._is_radial(product_code)
        data_block = f.sym_block[0][0]
        data = np.ma.array(data_block["data"])
        if self.dtype == "VEL":
            mapped_data = np.ma.masked_invalid(velocity_mapper(data))
            rf = np.ma.masked_not_equal(mapped_data, 30)
            data = np.ma.masked_equal(mapped_data, 30)
            self.data = (data, rf)
        else:
            data[data == 0] = np.ma.masked
            self.data = np.ma.masked_invalid(f.map_data(data))
        if self.dtype == "ET":
            # convert kft to km
            self.data *= 0.30478
        station_info = _get_radar_info(self.code)
        self.radar_type = station_info[3]
        self.max_range = int(f.max_range)
        # Hard coding to adjust max range for different types of radar
        if f.max_range >= 230:
            if self.radar_type in ["SC", "CC"]:
                self.max_range = 150
            elif self.radar_type in ["CA", "CB"]:
                self.max_range = 200
            elif self.radar_type == "CD":
                self.max_range = 125
        if self.radial_flag:
            self.az = np.array(data_block["start_az"]) * deg2rad
            self.reso = self.max_range / data.shape[1]
            self.rng = np.arange(self.reso, self.max_range + self.reso, self.reso)
        else:
            xdim, ydim = data.shape
            x = np.linspace(self.max_range * -1, self.max_range, xdim) / 111 + f.lon
            y = np.linspace(self.max_range, self.max_range * -1, ydim) / 111 + f.lat
            self.lon, self.lat = np.meshgrid(x, y)
            self.reso = self.max_range / data.shape[0] * 2
        self.stationlat = f.lat
        self.stationlon = f.lon
        self.el = np.round_(f.metadata["el_angle"], 1)
        self.scantime = f.metadata["vol_time"]

    def get_data(self) -> Dataset:
        if self.radial_flag:
            lon, lat = self.projection()
            if self.dtype in ["VEL", "SW"]:
                da = DataArray(
                    self.data[0],
                    coords=[self.az, self.rng],
                    dims=["azimuth", "distance"],
                )
            else:
                da = DataArray(
                    self.data, coords=[self.az, self.rng], dims=["azimuth", "distance"]
                )
            ds = Dataset(
                {self.dtype: da},
                attrs={
                    "elevation": self.el,
                    "range": self.max_range,
                    "scan_time": self.scantime,
                    "site_code": self.code,
                    "site_name": self.name,
                    "site_longitude": self.stationlon,
                    "site_latitude": self.stationlat,
                    "tangential_reso": self.reso,
                },
            )
            ds["longitude"] = (["azimuth", "distance"], lon)
            ds["latitude"] = (["azimuth", "distance"], lat)
            if self.dtype in ["VEL", "SW"]:
                ds["RF"] = (["azimuth", "distance"], self.data[1])
        else:
            da = DataArray(
                self.data,
                coords=[self.lon[0], self.lat[:, 0]],
                dims=["longitude", "latitude"],
            )
            ds = Dataset(
                {self.dtype: da},
                attrs={
                    "elevation": 0,
                    "range": self.max_range,
                    "scan_time": self.scantime,
                    "site_code": self.code,
                    "site_name": self.name,
                    "site_longitude": self.stationlon,
                    "site_latitude": self.stationlat,
                    "tangential_reso": self.reso,
                },
            )
        return ds

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
        elif spec in range(37, 39):
            return "CR"
        elif spec == 41:
            return "ET"
        elif spec == 57:
            return "VIL"
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

    def get_data(self, level=0) -> Dataset:
        dtype = self.product_name
        if self.data.ndim == 2:
            ret = self.data
        else:
            ret = self.data[level]
            if self.product_name == "3DREF":
                dtype = "CR"
        da = DataArray(ret, coords=[self.lon, self.lat], dims=["longitude", "latitude"])
        ds = Dataset(
            {dtype: da},
            attrs={
                "scan_time": self.scantime,
                "site_code": "SWAN",
                "site_name": "SWAN",
                "tangential_reso": np.nan,
            },
        )
        return ds


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
        da = DataArray(self.data, coords=[self.azi, dist], dims=["azimuth", "distance"])
        ds = Dataset(
            {dtype: da},
            attrs={
                "elevation": self.el,
                "range": self.end_range,
                "scan_time": self.scantime,
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": self.reso,
                "task": self.task_name,
            },
        )
        ds["longitude"] = (["azimuth", "distance"], lon)
        ds["latitude"] = (["azimuth", "distance"], lat)
        ds["height"] = (["azimuth", "distance"], hgt)
        if dtype in ["VEL", "SW"]:
            ds["RF"] = (["azimuth", "distance"], self.data[1])
        return radial
