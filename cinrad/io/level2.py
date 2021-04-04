# -*- coding: utf-8 -*-
# Author: Puyuan Du

import warnings
import datetime
from pathlib import Path
from collections import namedtuple, defaultdict
from typing import Union, Optional, List, Any, Generator

import numpy as np
import xarray as xr

from cinrad.constants import deg2rad, con
from cinrad.projection import get_coordinate, height
from cinrad.error import RadarDecodeError
from cinrad.io.base import RadarBase, prepare_file
from cinrad.io._dtype import *
from cinrad._typing import Number_T

__all__ = ["CinradReader", "StandardData"]

ScanConfig = namedtuple("ScanConfig", SDD_cut.fields.keys())
utc_offset = datetime.timedelta(hours=8)


def vcp(el_num: int) -> str:
    r"""Determine volume coverage pattern by number of scans."""
    if el_num == 5:
        task_name = "VCP31"
    elif el_num == 9:
        task_name = "VCP21"
    elif el_num == 14:
        task_name = "VCP11"
    else:
        task_name = "Unknown"
    return task_name


def infer_type(f: Any, filename: str) -> tuple:
    r"""Detect radar type from records in file"""
    # Attempt to find information in file, which has higher
    # priority compared with information obtained from file name
    radartype = None
    code = None
    f.seek(100)
    typestring = f.read(9)
    if typestring == b"CINRAD/SC":
        radartype = "SC"
    elif typestring == b"CINRAD/CD":
        radartype = "CD"
    f.seek(116)
    if f.read(9) == b"CINRAD/CC":
        radartype = "CC"
    # Read information from filename (if applicable)
    if filename.startswith("RADA"):
        spart = filename.split("-")
        if len(spart) > 2:
            code = spart[1]
            radartype = spart[2]
    elif filename.startswith("Z"):
        spart = filename.split("_")
        if len(spart) > 7:
            code = spart[3]
            radartype = spart[7]
    return code, radartype


class CinradReader(RadarBase):
    r"""
    Class reading old-version CINRAD data.

    Args:
        file (str, IO): Path points to the file or a file object.

        radar_type (str): Type of the radar.

        file_name (str): Name of the file, only used when `file` argument is
        a file object.
    """

    def __init__(
        self, file: Any, radar_type: Optional[str] = None,
    ):
        f = prepare_file(file)
        filename = Path(file).name if isinstance(file, str) else ""
        self.code, t_infer = infer_type(f, filename)
        if radar_type:
            if t_infer != radar_type:
                warnings.warn(
                    "Contradictory information from input radar type and"
                    "radar type detected from input file."
                )
            self.radartype = radar_type
        else:
            if not t_infer:
                raise RadarDecodeError(
                    "Unable to determine the file type. Use `radar_type` keyword"
                    "to specify the radar type."
                )
            self.radartype = t_infer
        self.site_info = {}
        f.seek(0)
        if self.radartype in ["SA", "SB"]:
            self._SAB_reader(f)
        elif self.radartype in ["CA", "CB"]:
            self._SAB_reader(f, dtype="CAB")
        elif self.radartype == "CC2":
            self._CC2_reader(f)
        else:
            try:
                if self.radartype == "CC":
                    self._CC_reader(f)
                elif self.radartype in ["SC", "CD"]:
                    self._CD_reader(f)
                else:
                    raise RadarDecodeError("Unrecognized data")
            except Exception as err:
                # Currently there's no good way to differentiate the special
                # SC/CC files, so catch the exception of normal decoding process
                # and try this one if possible
                try:
                    f.seek(0)
                    self._SAB_reader(f, dtype="special")
                except:
                    raise err
        self._update_radar_info()
        # TODO: Override information
        if "longitude" in self.site_info:
            self.stationlon = self.site_info["longitude"]
        if "latitude" in self.site_info:
            self.stationlat = self.site_info["latitude"]
        if "height" in self.site_info:
            self.radarheight = self.site_info["height"]
        if "name" in self.site_info:
            self.name = self.site_info["name"]
        if "code" in self.site_info:
            self.code = self.site_info["code"]
        if self.code == None and self.name:
            # Use name as code when code is missing
            self.code = self.name
        f.close()

    def _SAB_reader(self, f: Any, dtype: str = "SAB"):
        _header_size = 128
        if dtype == "SAB":
            radar_dtype = SAB_dtype
        elif dtype == "CAB":
            radar_dtype = CAB_dtype
        elif dtype == "special":
            radar_dtype = S_SPECIAL_dtype
            _header_size = 132
        data = np.frombuffer(f.read(), dtype=radar_dtype)
        start = datetime.datetime(1969, 12, 31)
        deltday = datetime.timedelta(days=int(data["day"][0]))
        deltsec = datetime.timedelta(milliseconds=int(data["time"][0]))
        self.scantime = start + deltday + deltsec
        self.Rreso = data["gate_length_r"][0] / 1000
        self.Vreso = data["gate_length_v"][0] / 1000
        boundary = np.where(data["radial_num"] == 1)[0]
        self.el = data["elevation"][boundary] * con
        self.azimuth = data["azimuth"] * con * deg2rad
        dv = data["v_reso"][0]
        self.nyquist_v = data["nyquist_vel"][boundary] / 100
        self.task_name = "VCP{}".format(data["vcp_mode"][0])
        f.seek(0)
        size = radar_dtype.itemsize
        b = np.append(boundary, data.shape[0])
        gnr = data["gate_num_r"][boundary]
        gnv = data["gate_num_v"][boundary]
        out_data = dict()
        for bidx, rnum, vnum, idx in zip(np.diff(b), gnr, gnv, range(len(b))):
            # `bidx`: number of data blocks (i.e. radials)
            # `rnum`: number of reflectivity gates.
            # `vnum`: number of velocity gates.
            # `idx`: number of scans.

            # Construct a temporary dtype to parse data more efficiently
            temp_dtype = [
                ("header", "u1", _header_size),
                ("ref", "u1", rnum),
                ("vel", "u1", vnum),
                ("sw", "u1", vnum),
                ("res", "u1", size - _header_size - rnum - vnum * 2),
            ]
            da = np.frombuffer(f.read(bidx * size), dtype=np.dtype(temp_dtype))
            out_data[idx] = dict()
            r = (np.ma.masked_equal(da["ref"], 0) - 2) / 2 - 32
            r[r == 95.5] = 0
            out_data[idx]["REF"] = r
            v = np.ma.masked_less(da["vel"], 2)
            sw = np.ma.masked_less(da["sw"], 2)
            if dv == 2:
                out_data[idx]["VEL"] = (v - 2) / 2 - 63.5
                out_data[idx]["SW"] = (sw - 2) / 2 - 63.5
            elif dv == 4:
                out_data[idx]["VEL"] = v - 2 - 127
                out_data[idx]["SW"] = sw - 2 - 127
            out_data[idx]["azimuth"] = self.azimuth[b[idx] : b[idx + 1]]
            out_data[idx]["RF"] = np.ma.masked_not_equal(da["vel"], 1)
        angleindex = np.arange(0, data["el_num"][-1], 1)
        if dtype == "special":
            self.angleindex_r = self.angleindex_v = angleindex
        else:
            self.angleindex_r = np.delete(angleindex, [1, 3])
            self.angleindex_v = np.delete(angleindex, [0, 2])
        self.data = out_data

    def _CC_reader(self, f: Any):
        header = np.frombuffer(f.read(1024), CC_header)
        self.site_info = {
            "name": header["cStation"][0].decode("gbk", "ignore"),
            "code": header["cStationNumber"][0].decode("utf-8", "ignore")[:5],
        }
        scan_mode = header["ucScanMode"][0]
        if scan_mode < 100:
            raise NotImplementedError("Only VPPI scan mode is supported")
        stop_angle = scan_mode - 100
        self.scantime = (
            datetime.datetime(
                header["ucEYear1"][0] * 100 + header["ucEYear2"][0],
                header["ucEMonth"][0],
                header["ucEDay"][0],
                header["ucEHour"][0],
                header["ucEMinute"][0],
                header["ucESecond"][0],
            )
            - utc_offset
        )
        f.seek(218)
        param = np.frombuffer(f.read(660), CC_param)
        self.el = param["usAngle"][:stop_angle] / 100
        self.nyquist_v = param["usMaxV"][:stop_angle] / 100
        self.task_name = vcp(len(self.el))
        f.seek(1024)
        data = np.frombuffer(f.read(), CC_data)
        r = np.ma.masked_equal(data["Z"], -0x8000) / 10
        v = np.ma.masked_equal(data["V"], -0x8000) / 10
        w = np.ma.masked_equal(data["W"], -0x8000) / 10
        self.Rreso = 0.3
        self.Vreso = 0.3
        data = dict()
        for i in range(len(self.el)):
            data[i] = dict()
            data[i]["REF"] = r[i * 512 : (i + 1) * 512]
            data[i]["VEL"] = v[i * 512 : (i + 1) * 512]
            data[i]["SW"] = w[i * 512 : (i + 1) * 512]
            data[i]["azimuth"] = self.get_azimuth_angles(i)
        self.data = data
        self.angleindex_r = self.angleindex_v = [i for i in range(len(self.el))]

    def _CD_reader(self, f: Any):
        header = np.frombuffer(f.read(CD_dtype.itemsize), CD_dtype)
        el_num = header["obs"]["stype"][0] - 100  # VOL
        self.task_name = vcp(el_num)
        self.scantime = (
            datetime.datetime(
                header["obs"]["syear"][0],
                header["obs"]["smonth"][0],
                header["obs"]["sday"][0],
                header["obs"]["shour"][0],
                header["obs"]["sminute"][0],
                header["obs"]["ssecond"][0],
            )
            - utc_offset
        )
        self.nyquist_v = header["obs"]["layerparam"]["MaxV"][0][:el_num] / 100
        self.Rreso = self.Vreso = 0.25
        self.el = header["obs"]["layerparam"]["Swangles"][0][:el_num] / 100
        data = dict()
        for el in range(el_num):
            full_scan = np.frombuffer(f.read(360 * CD_DATA.itemsize), CD_DATA)
            # Avoid uint8 arithmetic overflow
            raw_ref = full_scan["rec"]["m_dbz"].astype(int)
            raw_vel = full_scan["rec"]["m_vel"].astype(int)
            raw_sw = full_scan["rec"]["m_sw"].astype(int)
            data[el] = dict()
            data[el]["REF"] = (np.ma.masked_equal(raw_ref, 0) - 64) / 2
            data[el]["VEL"] = (
                self.nyquist_v[el] * (np.ma.masked_equal(raw_vel, 0) - 128) / 128
            )
            data[el]["SW"] = self.nyquist_v[el] * np.ma.masked_equal(raw_sw, 0) / 256
            data[el]["RF"] = np.ma.masked_not_equal(raw_vel, 128)
        self.data = data
        self.angleindex_r = self.angleindex_v = list(range(el_num))

    def _CC2_reader(self, f: Any):
        header = np.frombuffer(f.read(CC2_header.itemsize), CC2_header)
        self.site_info = {
            "longitude": header["lLongitudeValue"][0] / 1000,
            "latitude": header["lLatitudeValue"][0] / 1000,
            "height": header["lHeight"][0] / 1000,
            "name": header["sStation"][0].decode(),
        }
        obs_param = np.frombuffer(f.read(CC2_obs.itemsize), CC2_obs)
        self.scantime = (
            datetime.datetime(
                obs_param["usSYear"][0],
                obs_param["ucSMonth"][0],
                obs_param["ucSDay"][0],
                obs_param["ucSHour"][0],
                obs_param["ucSMinute"][0],
                obs_param["ucSSecond"][0],
            )
            - utc_offset
        )
        layer_info = obs_param["LayerInfo"]
        scan_type = obs_param["ucType"][0]
        if scan_type == 1:
            # RHI
            raise NotImplementedError("RHI scan type is not supported")
        elif scan_type == 10:
            # PPI
            raise NotImplementedError("Single layer PPI scan type is not supported")
        elif 100 < scan_type < 200:
            # VOL
            el_num = scan_type - 100
            radial_num = layer_info["usRecordNumber"][0][:el_num]
            self.el = layer_info["sSwpAngle"][0][:el_num] / 100
            self.nyquist_v = layer_info["usMaxV"][0][:el_num] / 100
            self.Rreso = layer_info["usZBinWidth"][0][0] / 10000
            self.Vreso = layer_info["usVBinWidth"][0][0] / 10000
            other_info = np.frombuffer(f.read(CC2_other.itemsize), CC2_other)
            self.task_name = other_info["sScanName"][0].decode()
            data_block = np.frombuffer(f.read(), CC2_data)
            raw_azimuth = data_block["usAzimuth"] / 100 * deg2rad
            raw_dbz = np.ma.masked_equal(data_block["ucCorZ"].astype(int), 0)
            dbz = (raw_dbz - 64) / 2
            zdr = np.ma.masked_equal(data_block["siZDR"].astype(int), -32768) * 0.01
            phi = np.ma.masked_equal(data_block["siPHDP"].astype(int), -32768) * 0.01
            rho = np.ma.masked_equal(data_block["siROHV"].astype(int), 0) * 0.001
            kdp = np.ma.masked_equal(data_block["uKDP"].astype(int), -32768) * 0.01
            data = dict()
            radial_idx = [0] + (np.cumsum(radial_num) - 1).tolist()
            for el in range(el_num):
                start_radial_idx = radial_idx[el]
                end_radial_idx = radial_idx[el + 1]
                data[el] = dict()
                data[el]["azimuth"] = raw_azimuth[start_radial_idx:end_radial_idx]
                data[el]["REF"] = dbz[start_radial_idx:end_radial_idx]
                data[el]["ZDR"] = zdr[start_radial_idx:end_radial_idx]
                data[el]["PHI"] = phi[start_radial_idx:end_radial_idx]
                data[el]["RHO"] = rho[start_radial_idx:end_radial_idx]
                data[el]["KDP"] = kdp[start_radial_idx:end_radial_idx]
            self.data = data
            self.angleindex_r = self.angleindex_v = list(range(el_num))

    def get_nrays(self, scan: int) -> int:
        r"""Get number of radials in certain scan"""
        if self.radartype in ["SA", "SB", "CA", "CB"]:
            return len(self.data[scan]["azimuth"])
        elif self.radartype == "CC":
            return 512
        elif self.radartype in ["SC", "CD"]:
            return 360

    def get_azimuth_angles(self, scans: Optional[int] = None) -> np.ndarray:
        r"""Get index of input azimuth angle (radian)"""
        if self.radartype in ["SA", "SB", "CA", "CB", "CC2"]:
            if scans is None:
                return self.azimuth
            else:
                return self.data[scans]["azimuth"]
        elif self.radartype == "CC":
            if scans is None:
                return (
                    np.array(np.linspace(0, 360, 512).tolist() * self.get_nscans())
                    * deg2rad
                )
            else:
                return np.array(np.linspace(0, 360, 512).tolist()) * deg2rad
        elif self.radartype in ["SC", "CD"]:
            if scans is None:
                return (
                    np.array(np.linspace(0, 360, 360).tolist() * self.get_nscans())
                    * deg2rad
                )
            else:
                return np.array(np.linspace(0, 360, 360).tolist()) * deg2rad

    def get_elevation_angles(
        self, scans: Optional[int] = None
    ) -> Union[np.ndarray, float]:
        if scans is None:
            return self.el
        else:
            return self.el[scans]

    def get_raw(
        self, tilt: int, drange: Number_T, dtype: str
    ) -> Union[np.ndarray, tuple]:
        r"""
        Get radar raw data

        Args:
            tilt (int): Index of elevation angle starting from zero.
            
            drange (float): Radius of data.

            dtype (str): Type of product (REF, VEL, etc.)

        Returns:
            numpy.ndarray or tuple of numpy.ndarray: Raw data
        """
        rf_flag = False
        self.tilt = tilt
        reso = self.Rreso if dtype == "REF" else self.Vreso
        dmax = np.round(self.data[tilt][dtype][0].shape[0] * reso)
        if dmax < drange:
            warnings.warn("Requested data range exceed max range in this tilt")
        self.drange = drange
        self.elev = self.el[tilt]
        try:
            data = np.ma.array(self.data[tilt][dtype])
            # The structure of `out_data`:
            # The key of `out_data` is the number of scan counting from zero (int).
            # The value of `out_data` is also dictionary, the key of it are the abbreviations of
            # product name, such as `REF`, `VEL`.
            # The value of this sub-dictionary is the data stored in np.ma.MaskedArray.
        except KeyError:
            raise RadarDecodeError("Invalid product name")
        ngates = int(drange // reso)
        cut = data.T[:ngates]
        shape_diff = ngates - cut.shape[0]
        append = np.zeros((int(np.round(shape_diff)), cut.shape[1])) * np.ma.masked
        if dtype in ["VEL", "SW"]:
            try:
                rf = self.data[tilt]["RF"]
            except KeyError:
                pass
            else:
                rf_flag = True
                rf = rf.T[:ngates]
                rf = np.ma.vstack([rf, append])
        r = np.ma.vstack([cut, append])
        if rf_flag:
            r.mask = np.logical_or(r.mask, ~rf.mask)
            ret = (r.T, rf.T)
        else:
            ret = r.T
        return ret

    def get_data(self, tilt: int, drange: Number_T, dtype: str) -> xr.Dataset:
        r"""
        Get radar data with extra information

        Args:
            tilt (int): Index of elevation angle starting from zero.
            
            drange (float): Radius of data.

            dtype (str): Type of product (REF, VEL, etc.)

        Returns:
            xarray.Dataset: Data.
        """
        task = getattr(self, "task_name", None)
        reso = self.Rreso if dtype == "REF" else self.Vreso
        ret = self.get_raw(tilt, drange, dtype)
        rf_flag = (dtype in ["VEL", "SW"]) and ("RF" in self.data[tilt])
        x, y, z, d, a = self.projection(reso)
        shape = ret[0].shape[1] if rf_flag else ret.shape[1]
        if rf_flag:
            da = xr.DataArray(ret[0], coords=[a, d], dims=["azimuth", "distance"])
        else:
            da = xr.DataArray(ret, coords=[a, d], dims=["azimuth", "distance"])
        ds = xr.Dataset(
            {dtype: da},
            attrs={
                "elevation": self.elev,
                "range": int(np.round(shape * reso)),
                "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                "site_code": self.code,
                "site_name": self.name,
                "site_longitude": self.stationlon,
                "site_latitude": self.stationlat,
                "tangential_reso": reso,
                "nyquist_vel": self.nyquist_v[tilt],
                "task": task,
            },
        )
        ds["longitude"] = (["azimuth", "distance"], x)
        ds["latitude"] = (["azimuth", "distance"], y)
        ds["height"] = (["azimuth", "distance"], z)
        if rf_flag:
            ds["RF"] = (["azimuth", "distance"], ret[1])
        return ds

    def projection(self, reso: float, h_offset: bool = False) -> tuple:
        r"""Calculate the geographic coordinates of the requested data range."""
        theta = self.get_azimuth_angles(self.tilt)
        r = self.get_range(self.drange, reso)
        lonx, latx = get_coordinate(
            r, theta, self.elev, self.stationlon, self.stationlat, h_offset=h_offset
        )
        hght = (
            height(r, self.elev, self.radarheight)
            * np.ones(theta.shape[0])[:, np.newaxis]
        )
        return lonx, latx, hght, r, theta

    def iter_tilt(self, drange: Number_T, dtype: str) -> Generator:
        if dtype == "REF":
            tlist = self.angleindex_r
        elif dtype in ["VEL", "SW"]:
            tlist = self.angleindex_v
        for i in tlist:
            yield self.get_data(i, drange, dtype)


class StandardData(RadarBase):
    r"""
    Class reading data in standard format.

    Args:
        file (str, IO): Path points to the file or a file object.
    """
    # fmt: off
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 32:'Zc', 33:'Vc', 34:'Wc', 35:'ZDRc'}
    # fmt: on
    def __init__(self, file: Any):
        with prepare_file(file) as self.f:
            self._parse()
        self._update_radar_info()
        # In standard data, station information stored in file
        # has higher priority, so we override some information.
        self.stationlat = self.geo["lat"]
        self.stationlon = self.geo["lon"]
        self.radarheight = self.geo["height"]
        if self.name == "None":
            # Last resort to find info
            self.name = self.geo["name"]
        self.angleindex_r = self.available_tilt("REF")  # API consistency
        del self.geo

    def _parse(self):
        header = np.frombuffer(self.f.read(32), SDD_header)
        if header["magic_number"] != 0x4D545352:
            raise RadarDecodeError("Invalid standard data")
        site_config = np.frombuffer(self.f.read(128), SDD_site)
        self.code = (
            site_config["site_code"][0]
            .decode("ascii", errors="ignore")
            .replace("\x00", "")
        )
        freq = site_config["frequency"][0]
        self.wavelength = 3e8 / freq / 10000
        self.geo = geo = dict()
        geo["lat"] = site_config["Latitude"][0]
        geo["lon"] = site_config["Longitude"][0]
        geo["height"] = site_config["ground_height"][0]
        geo["name"] = site_config["site_name"][0].decode("ascii", errors="ignore")
        task = np.frombuffer(self.f.read(256), SDD_task)
        self.task_name = (
            task["task_name"][0].decode("ascii", errors="ignore").split("\x00")[0]
        )
        self.scantime = datetime.datetime(1970, 1, 1) + datetime.timedelta(
            seconds=int(task["scan_start_time"])
        )
        cut_num = task["cut_number"][0]
        scan_config = np.frombuffer(self.f.read(256 * cut_num), SDD_cut)
        self.scan_config = [ScanConfig(*i) for i in scan_config]
        # TODO: improve repr
        data = dict()
        # `aux` stores some auxiliary information, including azimuth angles, elevation angles,
        # and scale and offset of data.
        aux = dict()
        if task["scan_type"] == 2:  # Single-layer RHI
            self.scan_type = "RHI"
        else:
            # There are actually some other scan types, however, they are not currently supported.
            self.scan_type = "PPI"
        # Some attributes that are used only for converting to pyart.core.Radar instances
        self._time_radial = list()
        self._sweep_start_ray_index = list()
        self._sweep_end_ray_index = list()
        # Time for each radial
        radial_count = 0
        while 1:
            header_bytes = self.f.read(64)
            if not header_bytes:
                # Fix for single-tilt file
                break
            radial_header = np.frombuffer(header_bytes, SDD_rad_header)
            if radial_header["zip_type"][0] == 1:  # LZO compression
                raise NotImplementedError("LZO compressed file is not supported")
            self._time_radial.append(
                radial_header["seconds"][0] + radial_header["microseconds"][0] / 1e6
            )
            el_num = radial_header["elevation_number"][0] - 1
            if el_num not in data.keys():
                data[el_num] = defaultdict(list)
                aux[el_num] = defaultdict(list)
            aux[el_num]["azimuth"].append(radial_header["azimuth"][0])
            aux[el_num]["elevation"].append(radial_header["elevation"][0])
            for _ in range(radial_header["moment_number"][0]):
                moment_header = np.frombuffer(self.f.read(32), SDD_mom_header)
                dtype_code = moment_header["data_type"][0]
                dtype = self.dtype_corr.get(dtype_code, None)
                data_body = np.frombuffer(
                    self.f.read(moment_header["block_length"][0]),
                    "u{}".format(moment_header["bin_length"][0]),
                )
                if not dtype:
                    warnings.warn(
                        "Data type {} not understood, skipping".format(dtype_code),
                        RuntimeWarning,
                    )
                    continue
                if dtype not in aux[el_num].keys():
                    scale = moment_header["scale"][0]
                    offset = moment_header["offset"][0]
                    aux[el_num][dtype] = (scale, offset)
                # In `StandardData`, the `data` dictionary stores raw data instead of data
                # calibrated by scale and offset.
                # The calibration process is moved to `get_raw` part.
                data[el_num][dtype].append(data_body)
            radial_state = radial_header["radial_state"][0]
            if radial_state in [0, 3]:
                # Start of tilt or volume scan
                self._sweep_start_ray_index.append(radial_count)
            elif radial_state in [2, 4]:
                self._sweep_end_ray_index.append(radial_count)
            radial_count += 1
            if radial_state in [4, 6]:  # End scan
                break
        self.data = data
        self.aux = aux
        self.el = [i.elev for i in self.scan_config]

    @classmethod
    def merge(cls, files: List[str], output: str):
        r"""
        Merge single-tilt standard data into a volumetric scan

        Args:
            files (List[str]): List of path of data to be merged

            output (str): The file path to store the merged data
        """
        with prepare_file(files[0]) as first_file:
            first_file.seek(160)
            task = np.frombuffer(first_file.read(256), SDD_task)
            cut_num = task["cut_number"][0]
            total_seek_bytes = first_file.tell() + 256 * cut_num
            all_tilt_data = [b""] * cut_num
            first_file.seek(0)
            header_bytes = first_file.read(total_seek_bytes)
            rad = np.frombuffer(first_file.read(64), SDD_rad_header)
            el_num = rad["elevation_number"][0] - 1
            first_file.seek(total_seek_bytes)
            all_tilt_data[el_num] = first_file.read()
            for f in files[1:]:
                with prepare_file(f) as buf:
                    buf.seek(total_seek_bytes)
                    rad = np.frombuffer(buf.read(64), SDD_rad_header)
                    buf.seek(total_seek_bytes)
                    el_num = rad["elevation_number"][0] - 1
                    all_tilt_data[el_num] = buf.read()
        with open(output, "wb") as out:
            out.write(header_bytes)
            out.write(b"".join(all_tilt_data))

    def get_raw(
        self, tilt: int, drange: Number_T, dtype: str
    ) -> Union[np.ndarray, tuple]:
        r"""
        Get radar raw data

        Args:
            tilt (int): Index of elevation angle starting from zero.
            
            drange (float): Radius of data.

            dtype (str): Type of product (REF, VEL, etc.)

        Returns:
            numpy.ndarray or tuple of numpy.ndarray: Raw data
        """
        # The scan number is set to zero in RHI mode.
        self.tilt = tilt if self.scan_type == "PPI" else 0
        self.drange = drange
        if self.scan_type == "RHI":
            max_range = self.scan_config[0].max_range1 / 1000
            if drange > max_range:
                drange = max_range
        self.elev = self.el[tilt]
        reso = self.scan_config[tilt].dop_reso / 1000
        try:
            raw = np.array(self.data[tilt][dtype])
        except KeyError:
            raise RadarDecodeError("Invalid product name")
        ngates = int(drange // reso)
        if raw.size == 0:
            warnings.warn("Empty data", RuntimeWarning)
            # Calculate size equivalent
            nrays = len(self.aux[tilt]["azimuth"])
            out = np.zeros((nrays, ngates)) * np.ma.masked
            return out
        # Data below 5 are used as reserved codes, which are used to indicate other
        # information instead of real data, so they should be masked.
        data = np.ma.masked_less(raw, 5)
        cut = data[:, :ngates]
        shape_diff = ngates - cut.shape[1]
        append = np.zeros((cut.shape[0], int(shape_diff))) * np.ma.masked
        if dtype in ["VEL", "SW"]:
            # The reserved code 1 indicates folded velocity.
            # These region will be shaded by color of `RF`.
            rf = np.ma.masked_not_equal(cut.data, 1)
            rf = np.ma.hstack([rf, append])
        cut = np.ma.hstack([cut, append])
        scale, offset = self.aux[tilt][dtype]
        r = (cut - offset) / scale
        if dtype in ["VEL", "SW"]:
            ret = (r, rf)
            # RF data is separately packed into the data.
        else:
            ret = r
        return ret

    def get_data(self, tilt: int, drange: Number_T, dtype: str) -> xr.Dataset:
        r"""
        Get radar data with extra information

        Args:
            tilt (int): Index of elevation angle starting from zero.
            
            drange (float): Radius of data.

            dtype (str): Type of product (REF, VEL, etc.)

        Returns:
            xarray.Dataset: Data.
        """
        reso = self.scan_config[tilt].dop_reso / 1000
        ret = self.get_raw(tilt, drange, dtype)
        shape = ret[0].shape[1] if isinstance(ret, tuple) else ret.shape[1]
        if self.scan_type == "PPI":
            x, y, z, d, a = self.projection(reso)
            if dtype in ["VEL", "SW"]:
                da = xr.DataArray(ret[0], coords=[a, d], dims=["azimuth", "distance"])
            else:
                da = xr.DataArray(ret, coords=[a, d], dims=["azimuth", "distance"])
            ds = xr.Dataset(
                {dtype: da},
                attrs={
                    "elevation": self.elev,
                    "range": int(np.round(shape * reso)),
                    "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                    "site_code": self.code,
                    "site_name": self.name,
                    "site_longitude": self.stationlon,
                    "site_latitude": self.stationlat,
                    "tangential_reso": reso,
                    "nyquist_vel": self.scan_config[tilt].nyquist_spd,
                    "task": self.task_name,
                },
            )
            ds["longitude"] = (["azimuth", "distance"], x)
            ds["latitude"] = (["azimuth", "distance"], y)
            ds["height"] = (["azimuth", "distance"], z)
            if dtype in ["VEL", "SW"]:
                ds["RF"] = (["azimuth", "distance"], ret[1])
        else:
            # Manual projection
            dist = np.linspace(reso, self.drange, ret.shape[1])
            azimuth = self.aux[tilt]["azimuth"][0]
            elev = self.aux[tilt]["elevation"]
            d, e = np.meshgrid(dist, elev)
            h = height(d, e, self.radarheight)
            if dtype in ["VEL", "SW"]:
                da = xr.DataArray(
                    ret[0], coords=[elev, dist], dims=["tilt", "distance"]
                )
            else:
                da = xr.DataArray(ret, coords=[elev, dist], dims=["tilt", "distance"])
            # Calculate the "start" and "end" of RHI scan
            # to facilitate tick labeling
            start_lon = self.stationlon
            start_lat = self.stationlat
            end_lon, end_lat = get_coordinate(
                drange, azimuth * deg2rad, 0, self.stationlon, self.stationlat
            )
            ds = xr.Dataset(
                {dtype: da},
                attrs={
                    "range": drange,
                    "scan_time": self.scantime.strftime("%Y-%m-%d %H:%M:%S"),
                    "site_code": self.code,
                    "site_name": self.name,
                    "site_longitude": self.stationlon,
                    "site_latitude": self.stationlat,
                    "tangential_reso": reso,
                    "azimuth": azimuth,
                    "start_lon": start_lon,
                    "start_lat": start_lat,
                    "end_lon": end_lon,
                    "end_lat": end_lat,
                },
            )
            ds["x_cor"] = (["tilt", "distance"], d)
            ds["y_cor"] = (["tilt", "distance"], h)
        return ds

    def projection(self, reso: float) -> tuple:
        r = self.get_range(self.drange, reso)
        theta = np.array(self.aux[self.tilt]["azimuth"]) * deg2rad
        lonx, latx = get_coordinate(
            r, theta, self.elev, self.stationlon, self.stationlat
        )
        hght = (
            height(r, self.elev, self.radarheight)
            * np.ones(theta.shape[0])[:, np.newaxis]
        )
        return lonx, latx, hght, r, theta

    def available_tilt(self, product: str) -> List[int]:
        r"""Get all available tilts for given product"""
        tilt = list()
        for i in list(self.data.keys()):
            if product in self.data[i].keys():
                tilt.append(i)
        return tilt

    def iter_tilt(self, drange: Number_T, dtype: str) -> Generator:
        for i in self.available_tilt(dtype):
            yield self.get_data(i, drange, dtype)
