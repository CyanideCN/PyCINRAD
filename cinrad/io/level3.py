# -*- coding: utf-8 -*-
# Author: Puyuan Du

from collections import OrderedDict, defaultdict
from typing import Optional, Union, Any
import datetime
from io import BytesIO

import numpy as np
from xarray import Dataset, DataArray

from cinrad.projection import get_coordinate, height
from cinrad.constants import deg2rad
from cinrad._typing import Boardcast_T
from cinrad.io.base import RadarBase, prepare_file, _get_radar_info
from cinrad.io._dtype import *
from cinrad.error import RadarDecodeError


__all__ = ["PUP", "SWAN", "StandardPUP", "StormTrackInfo", "HailIndex"]


def xy2polar(x: Boardcast_T, y: Boardcast_T) -> tuple:
    return np.sqrt(x**2 + y**2), np.arctan2(x, y) * 180 / np.pi


# As metpy use a different table to map the data
# The color table constructed from PUP software is used to replace metpy
# fmt: off
velocity_tbl = np.array([np.nan, -27.5, -20.5, -15.5, -10.5, -5.5, -1.5,
                         -0.5, 0.5, 4.5, 9.5, 14.5, 19.5, 26.5, 27.5, 30])
# fmt: on


class PUP(RadarBase):
    r"""
    Class handling PUP data (Nexrad Level III data)

    Args:
        file (str, IO): Path points to the file or a file object.
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
        data = np.array(data_block["data"], dtype=int)
        if self.dtype == "VEL":
            mapped_data = np.ma.masked_invalid(velocity_tbl[data])
            rf = np.ma.masked_not_equal(mapped_data, 30)
            data = np.ma.masked_equal(mapped_data, 30)
            self.data = (data, rf)
        else:
            data[data == 0] = np.ma.masked
            self.data = np.ma.masked_invalid(f.map_data(data))
            if self.dtype == "ET":
                # convert kft to km
                self.data *= 0.30478
            elif self.dtype == "OHP":
                # convert in to mm
                self.data *= 25.4
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
            start_az = data_block["start_az"][0]
            az = np.linspace(0, 360, data.shape[0])
            az += start_az
            az[az > 360] -= 360
            self.az = az * deg2rad
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
        self.el = np.round_(f.metadata.get("el_angle", 0), 1)
        self.scantime = f.metadata["vol_time"]

    def get_data(self) -> Dataset:
        r"""
        Get radar data with extra information.

        Returns:
            xarray.Dataset: Data.
        """
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
                    "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
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
                coords=[self.lat[:, 0], self.lon[0]],
                dims=["latitude", "longitude"],
            )
            ds = Dataset(
                {self.dtype: da},
                attrs={
                    "elevation": 0,
                    "range": self.max_range,
                    "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
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
        return code in range(16, 31) or code == 78

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
        elif spec == 78:
            return "OHP"
        else:
            raise RadarDecodeError("Unsupported product type {}".format(spec))


class SWAN(object):
    r"""
    Class reading SWAN grid data.

    Args:
        file (str, IO): Path points to the file or a file object.
    """
    dtype_conv = {0: "B", 1: "b", 2: "u2", 3: "i2", 4: "u2"}
    size_conv = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2}

    def __init__(self, file: Any, product: Optional[str] = None):
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
            out = data_body.reshape(ydim, xdim)
        else:
            # 3D data
            out = data_body.reshape(zdim, ydim, xdim)
        self.data_time = datetime.datetime(
            header["year"][0],
            header["month"][0],
            header["day"][0],
            header["hour"][0],
            header["minute"][0],
        )
        # TODO: Recognize correct product name
        product_name = (
            (b"".join(header["data_name"]).decode("gbk", "ignore").replace("\x00", ""))
            if not product
            else product
        )
        self.product_name = product_name
        for pname in ["CR", "3DREF"]:
            if product_name.startswith(pname):
                self.product_name = pname
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

    def get_data(self, level: int = 0) -> Dataset:
        r"""
        Get radar data with extra information

        Args:
            level (int): The level of reflectivity data. Only used in `3DREF` data.

        Returns:
            xarray.Dataset: Data.
        """
        dtype = self.product_name
        if self.data.ndim == 2:
            ret = self.data
        else:
            ret = self.data[level]
            if self.product_name == "3DREF":
                dtype = "CR"
        da = DataArray(ret, coords=[self.lat, self.lon], dims=["latitude", "longitude"])
        ds = Dataset(
            {dtype: da},
            attrs={
                "scan_time": self.data_time.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": "SWAN",
                "site_name": "SWAN",
                "tangential_reso": np.nan,
                "range": np.nan,
                "elevation": 0,
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

    def track(self, storm_id: str, tracktype: str) -> Optional[tuple]:
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


class ProductParamsParser(object):
    @staticmethod
    def _ppi(buf):
        params = {}
        elev = np.frombuffer(buf.read(4), "f4")[0]
        params["elevation"] = elev
        return params

    @staticmethod
    def _rhi(buf):
        params = {}
        azi = np.frombuffer(buf.read(4), "f4")[0]
        top = np.frombuffer(buf.read(4), "i4")[0]
        bot = np.frombuffer(buf.read(4), "i4")[0]
        params["azimuth"] = azi
        params["top"] = top
        params["bottom"] = bot
        return params

    @staticmethod
    def _cappi(buf):
        params = {}
        layers_count = np.frombuffer(buf.read(4), "i4")[0]
        top = np.frombuffer(buf.read(4), "i4")[0]
        bot = np.frombuffer(buf.read(4), "i4")[0]
        filled = np.frombuffer(buf.read(4), "i4")[0]
        params["layers_count"] = layers_count
        params["top"] = top
        params["bottom"] = bot
        params["filled"] = filled
        return params

    @staticmethod
    def _empty(buf):
        pass

    @classmethod
    def parse(cls, product_type, param_bytes):
        buf = BytesIO(param_bytes)
        map_func = {
            1: cls._ppi,
            2: cls._rhi,
            3: cls._cappi,
            51: cls._ppi,
            52: cls._ppi,
            18: cls._empty,
        }
        params = {"elevation": 0}
        if product_type in map_func:
            params = map_func[product_type](buf)
        buf.close()
        return params


class StandardPUP(RadarBase):
    # fmt: off
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 32:'Zc', 33:'Vc', 34:'Wc', 35:'ZDRc', 36:'PDP', 37:'KDP',
                  38:'RHO', 71:'RR', 72:'HGT', 73:'VIL', 74:'SHR', 75:'RAIN', 76:'RMS',
                  77:'CTR'}

    ptype_corr = {1:"PPI", 2:"RHI", 3:"CAPPI", 4:"MAX", 6:"ET", 8:"VCS",
                  9:"LRA", 10:"LRM", 13:"SRR", 14:"SRM", 20:"WER", 23:"VIL",
                  24:"HSR", 25:"OHP", 26:"THP", 27:"STP", 28:"USP", 31:"VAD",
                  32:"VWP", 34:"Shear", 36:"SWP", 37:"STI", 38:"HI", 39:"M",
                  40:"TVS", 41:"SS", 48:"GAGE", 51:"HCL", 52:"QPE", 18:"CR",
                  44:"UAM"}
    # fmt: on
    def __init__(self, file):
        self.f = prepare_file(file)
        self._parse_header()
        self._update_radar_info()
        self.stationlat = self.geo["lat"][0]
        self.stationlon = self.geo["lon"][0]
        self.radarheight = self.geo["height"][0]
        if self.name == "None":
            self.name = self.code
        del self.geo
        if self.ptype in [1, 13, 14, 25, 26, 27, 28, 51, 52]:  # PPI radial format
            self._parse_radial_fmt()
        elif self.ptype in [6, 8, 9, 10, 18, 23]:
            self._parse_raster_fmt()
        elif self.ptype == 3:
            self._parse_cappi_fmt()
        elif self.ptype == 32:
            self._parse_vwp_fmt()
        elif self.ptype == 36:
            self._parse_swp_fmt()
        elif self.ptype == 37:
            self._parse_sti_fmt()
        elif self.ptype == 38:
            self._parse_hail_fmt()
        elif self.ptype == 39:
            self._parse_meso_fmt()
        elif self.ptype == 40:
            self._parse_tvs_fmt()
        elif self.ptype == 44:
            self._parse_uam_fmt()
        self.f.close()

    def _parse_header(self):
        header = np.frombuffer(self.f.read(32), SDD_header)
        if header["magic_number"] != 0x4D545352:
            raise RadarDecodeError("Invalid standard data")
        site_config = np.frombuffer(self.f.read(128), SDD_site)
        self.code = site_config["site_code"][0].decode().replace("\x00", "")
        self.geo = geo = dict()
        geo["lat"] = site_config["Latitude"]
        geo["lon"] = site_config["Longitude"]
        geo["height"] = site_config["ground_height"]
        task = np.frombuffer(self.f.read(256), SDD_task)
        self.task_name = task["task_name"][0].decode().replace("\x00", "")
        cut_num = task["cut_number"][0]
        self.scan_config = np.frombuffer(self.f.read(256 * cut_num), SDD_cut)
        ph = np.frombuffer(self.f.read(128), SDD_pheader)
        self.ptype = ph["product_type"][0]
        self.pname = self.ptype_corr[self.ptype]
        self.scantime = datetime.datetime.utcfromtimestamp(ph["scan_start_time"][0])
        if self.ptype == 1:
            self.pname = self.dtype_corr[ph["dtype_1"][0]]
        self.params = ProductParamsParser.parse(self.ptype, self.f.read(64))

    def _parse_radial_fmt(self):
        radial_header = np.frombuffer(self.f.read(64), L3_radial)
        bin_length = radial_header["bin_length"][0]
        scale = radial_header["scale"][0]
        offset = radial_header["offset"][0]
        reso = radial_header["reso"][0] / 1000
        start_range = radial_header["start_range"][0] / 1000
        end_range = radial_header["max_range"][0] / 1000
        nradial = radial_header["nradial"][0]
        data = list()
        azi = list()
        for _ in range(nradial):
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
        data_rf = np.ma.masked_not_equal(raw, 1)
        raw = np.ma.masked_less(raw, 5)
        data = (raw - offset) / scale
        if self.ptype in [25, 26, 27, 28]:
            # Mask 0 value in precipitation products
            data = np.ma.masked_equal(data, 0)
        az = np.linspace(0, 360, raw.shape[0])
        az += azi[0]
        az[az > 360] -= 360
        azi = az * deg2rad
        # self.azi = np.deg2rad(azi)
        dist = np.arange(start_range + reso, end_range + reso, reso)
        lon, lat = get_coordinate(
            dist, azi, self.params["elevation"], self.stationlon, self.stationlat
        )
        hgt = (
            height(dist, self.params["elevation"], self.radarheight)
            * np.ones(azi.shape[0])[:, np.newaxis]
        )
        da = DataArray(data, coords=[azi, dist], dims=["azimuth", "distance"])
        ds = Dataset(
            {self.pname: da},
            attrs={
                "elevation": self.params["elevation"],
                "range": end_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
                "task": self.task_name,
            },
        )
        ds["longitude"] = (["azimuth", "distance"], lon)
        ds["latitude"] = (["azimuth", "distance"], lat)
        ds["height"] = (["azimuth", "distance"], hgt)
        if self.pname in ["VEL", "SW"]:
            ds["RF"] = (["azimuth", "distance"], data_rf)
        self._dataset = ds

    def _parse_raster_fmt(self):
        raster_header = np.frombuffer(self.f.read(64), L3_raster)
        bin_length = raster_header["bin_length"][0]
        scale = raster_header["scale"][0]
        offset = raster_header["offset"][0]
        reso = raster_header["row_reso"][0] / 1000
        nx = raster_header["row_side_length"][0]
        ny = raster_header["col_side_length"][0]
        raw = (
            np.frombuffer(self.f.read(nx * ny * bin_length), "u{}".format(bin_length))
            .reshape(nx, ny)
            .astype(int)
        )
        raw = np.ma.masked_less(raw, 5)
        data = (raw - offset) / scale
        max_range = int(nx / 2 * reso)
        y = np.linspace(max_range, max_range * -1, ny) / 111 + self.stationlat
        x = (
            np.linspace(max_range * -1, max_range, nx) / (111 * np.cos(y * deg2rad))
            + self.stationlon
        )
        lon, lat = np.meshgrid(x, y)
        da = DataArray(
            data,
            coords=[lat[:, 0], lon[0]],
            dims=["latitude", "longitude"],
        )
        ds = Dataset(
            {self.pname: da},
            attrs={
                "elevation": 0,
                "range": max_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
            },
        )
        self._dataset = ds

    def _parse_cappi_fmt(self):
        azi = list()
        dist = list()
        data = list()
        height = np.linspace(
            self.params["bottom"], self.params["top"], self.params["layers_count"]
        )
        for _ in range(self.params["layers_count"]):
            radial_header = np.frombuffer(self.f.read(64), L3_radial)
            bin_length = radial_header["bin_length"][0]
            scale = radial_header["scale"][0]
            offset = radial_header["offset"][0]
            reso = radial_header["reso"][0] / 1000
            start_range = radial_header["start_range"][0] / 1000
            end_range = radial_header["max_range"][0] / 1000
            nradial = radial_header["nradial"][0]
            azi0 = list()
            for _ in range(nradial):
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
                azi0.append(start_a)
            if len(azi) == 0:
                raw = np.vstack(data).astype(int)
                az = np.linspace(0, 360, raw.shape[0])
                az += azi0[0]
                az[az > 360] -= 360
                azi = az * deg2rad
                dist = np.arange(start_range + reso, end_range + reso, reso)
        raw = np.vstack(data).astype(int)
        raw = np.ma.masked_less(raw, 5)
        data = (raw - offset) / scale
        data = np.reshape(data, (self.params["layers_count"], len(azi), len(dist)))
        lon, lat = get_coordinate(dist, azi, 0, self.stationlon, self.stationlat)
        da = DataArray(
            data, coords=[height, azi, dist], dims=["height", "azimuth", "distance"]
        )
        ds = Dataset(
            {self.pname: da},
            attrs={
                "elevation": 0,
                "range": end_range,
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
                "task": self.task_name,
            },
        )
        ds["longitude"] = (["azimuth", "distance"], lon)
        ds["latitude"] = (["azimuth", "distance"], lat)
        self._dataset = ds

    def _parse_vwp_fmt(self):
        vwp_header = np.frombuffer(self.f.read(32), L3_vwp_header)
        vwp_count = vwp_header["number_of_vols"][0]
        vwps = np.frombuffer(self.f.read(32 * vwp_count), L3_vwp)
        height = DataArray(vwps["height"])
        wind_direction = DataArray(vwps["wind_direction"])
        wind_speed = DataArray(vwps["wind_speed"])
        ds = Dataset(
            {
                "wind_direction": wind_direction,
                "wind_speed": wind_speed,
                "height": height,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
            },
        )
        self._dataset = ds

    def _parse_swp_fmt(self):
        swp_count = np.frombuffer(self.f.read(4), "i4")[0]
        swp = np.frombuffer(self.f.read(swp_count * 12), L3_swp)
        swp_azimuth = np.array(swp["azimuth"])
        swp_range = np.array(swp["range"])[:, np.newaxis]
        swp_percent = DataArray(swp["swp"])
        lon, lat = get_coordinate(
            swp_range / 1000,
            swp_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        ds = Dataset(
            {
                "swp_percent": swp_percent,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
            },
        )
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_hail_fmt(self):
        hail_count = np.frombuffer(self.f.read(4), "i4")[0]
        hail_table = np.frombuffer(self.f.read(hail_count * 28), L3_hail)
        ht0msl = np.frombuffer(self.f.read(4), "f4")[0]
        ht20msl = np.frombuffer(self.f.read(4), "f4")[0]
        hail_azimuth = np.array(hail_table["hail_azimuth"])
        hail_range = np.array(hail_table["hail_range"])[:, np.newaxis]
        hail_size = DataArray(hail_table["hail_size"])
        hail_possibility = DataArray(hail_table["hail_possibility"])
        hail_severe_possibility = DataArray(hail_table["hail_severe_possibility"])
        lon, lat = get_coordinate(
            hail_range / 1000,
            hail_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        ds = Dataset(
            {
                "hail_possibility": hail_possibility,
                "hail_size": hail_size,
                "hail_severe_possibility": hail_severe_possibility,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
                "height_0deg": ht0msl,
                "height_-20deg": ht20msl,
            },
        )
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_meso_fmt(self):
        storm_count = np.frombuffer(self.f.read(4), "i4")[0]
        meso_count = np.frombuffer(self.f.read(4), "i4")[0]
        feature_count = np.frombuffer(self.f.read(4), "i4")[0]
        meso_table = np.frombuffer(self.f.read(meso_count * 68), L3_meso)
        feature_table = np.frombuffer(self.f.read(feature_count * 72), L3_feature)
        npvthr = np.frombuffer(self.f.read(4), "i4")[0]
        fhthr = np.frombuffer(self.f.read(4), "f4")[0]
        meso_azimuth = np.array(meso_table["meso_azimuth"])
        meso_range = np.array(meso_table["meso_range"])[:, np.newaxis]
        lon, lat = get_coordinate(
            meso_range / 1000,
            meso_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )

        data_dict = {}
        # fmt: off
        for key in ["feature_id", "storm_id", "meso_azimuth", "meso_range", "meso_elevation",
                    "meso_avgshr", "meso_height", "meso_azdia", "meso_radius", "meso_avgrv",
                    "meso_mxrv", "meso_top", "meso_base", "meso_baseazim", "meso_baserange",
                    "meso_baseelevation", "meso_mxtanshr"]:
            data_dict[key] = DataArray(meso_table[key])
        # fmt: on
        attrs_dict = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            "npvthr": npvthr,
            "fhthr": fhthr,
        }
        ds = Dataset(data_dict, attrs=attrs_dict)
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_tvs_fmt(self):
        tvs_count = np.frombuffer(self.f.read(4), "i4")[0]
        etvs_count = np.frombuffer(self.f.read(4), "i4")[0]
        tvs_table = np.frombuffer(self.f.read((tvs_count + etvs_count) * 56), L3_tvs)
        minrefl = np.frombuffer(self.f.read(4), "i4")[0]
        minpvdv = np.frombuffer(self.f.read(4), "i4")[0]
        tvs_azimuth = np.array(tvs_table["tvs_azimuth"])
        tvs_range = np.array(tvs_table["tvs_range"])[:, np.newaxis]
        lon, lat = get_coordinate(
            tvs_range / 1000,
            tvs_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )

        data_dict = {}
        # fmt: off
        for key in ["tvs_id", "tvs_stormtype", "tvs_azimuth", "tvs_range", "tvs_elevation",
                    "tvs_lldv", "tvs_avgdv", "tvs_mxdv", "tvs_mxdvhgt", "tvs_depth", "tvs_base",
                    "tvs_top", "tvs_mxshr", "tvs_mxshrhgt"]:
            data_dict[key] = DataArray(tvs_table[key])
        # fmt: on
        attrs_dict = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            "minrefl": minrefl,
            "minpvdv": minpvdv,
        }
        ds = Dataset(data_dict, attrs=attrs_dict)
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def _parse_sti_fmt(self):
        sti_header = np.frombuffer(self.f.read(20), L3_sti_header)
        sti_count = sti_header["num_of_storms"][0]
        track_count = sti_count if sti_count < 100 else 100
        sti_current = np.frombuffer(self.f.read(24 * track_count), L3_sti_motion)
        curr_azimuth = np.array(sti_current["azimuth"])
        curr_range = np.array(sti_current["range"])[:, np.newaxis]
        curr_speed = sti_current["speed"]
        curr_direction = sti_current["direction"]
        curr_lon, curr_lat = get_coordinate(
            curr_range / 1000,
            curr_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        curr = [
            [curr_lon[i, 0], curr_lat[i, 0], curr_speed[i], curr_direction[i]]
            for i in range(track_count)
        ]
        forecast = []
        for _ in range(track_count):
            forecast_positon_count = np.frombuffer(self.f.read(4), "i4")[0]
            forecast_positon = np.frombuffer(
                self.f.read(12 * forecast_positon_count), L3_sti_position
            )
            fore_azimuth = np.array(forecast_positon["azimuth"])
            fore_range = np.array(forecast_positon["range"])[:, np.newaxis]
            fore_lon, fore_lat = get_coordinate(
                fore_range / 1000,
                fore_azimuth * deg2rad,
                self.params["elevation"],
                self.stationlon,
                self.stationlat,
            )
            fore = [
                [fore_lon[i, 0], fore_lat[i, 0]] for i in range(forecast_positon_count)
            ]
            forecast.append(fore)
        history = []
        for _ in range(track_count):
            history_positon_count = np.frombuffer(self.f.read(4), "i4")[0]
            history_positon = np.frombuffer(
                self.f.read(12 * history_positon_count), L3_sti_position
            )
            his_azimuth = np.array(history_positon["azimuth"])
            his_range = np.array(history_positon["range"])[:, np.newaxis]
            his_lon, his_lat = get_coordinate(
                his_range / 1000,
                his_azimuth * deg2rad,
                self.params["elevation"],
                self.stationlon,
                self.stationlat,
            )
            his = [[his_lon[i, 0], his_lat[i, 0]] for i in range(history_positon_count)]
            history.append(his)
        if sti_count > 0:
            self.sti_attributes = np.frombuffer(self.f.read(60), L3_sti_attribute)
            self.sti_components = np.frombuffer(self.f.read(12), L3_sti_component)
            self.sti_adaptation = np.frombuffer(self.f.read(40), L3_sti_adaptation)
        sti_data = [
            [{"curr": c, "fore": f, "his": h}]
            for c, f, h in zip(curr, forecast, history)
        ]
        attrs = {
            "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
            "site_code": self.code,
            "site_name": self.name,
            "site_longitude": self.stationlon,
            "site_latitude": self.stationlat,
            "task": self.task_name,
            "sti_count": sti_count,
        }
        ds = {"data": sti_data, "attrs": attrs}
        self._dataset = ds

    def _parse_uam_fmt(self):
        uam_count = np.frombuffer(self.f.read(4), "i4")[0]
        uam_table = np.frombuffer(self.f.read(uam_count * 44), L3_uam)
        uam_azimuth = np.array(uam_table["azimuth"])
        uam_range = np.array(uam_table["range"])[:, np.newaxis]
        uam_a = DataArray(uam_table["a"])
        uam_b = DataArray(uam_table["b"])
        uam_deg = DataArray(uam_table["deg"])
        lon, lat = get_coordinate(
            uam_range / 1000,
            uam_azimuth * deg2rad,
            self.params["elevation"],
            self.stationlon,
            self.stationlat,
        )
        ds = Dataset(
            {
                "a": uam_a,
                "b": uam_b,
                "deg": uam_deg,
            },
            attrs={
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "task": self.task_name,
            },
        )
        ds["longitude"] = DataArray(lon[:, 0])
        ds["latitude"] = DataArray(lat[:, 0])
        self._dataset = ds

    def get_data(self):
        return self._dataset
