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


def merge_bytes(byte_list: List[bytes]) -> bytes:
    return b"".join(byte_list)


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


def _detect_radartype(
    f: Any, filename: str, type_assert: Optional[str] = None
) -> tuple:
    r"""Detect radar type from records in file"""
    # Attempt to find information in file, which has higher
    # priority compared with information obtained from file name
    f.seek(100)
    typestring = f.read(9)
    det_sc = typestring == b"CINRAD/SC"
    det_cd = typestring == b"CINRAD/CD"
    f.seek(116)
    det_cc = f.read(9) == b"CINRAD/CC"
    # Read information from filename (if applicable)
    if filename.startswith("RADA"):
        spart = filename.split("-")
        code = spart[1]
        radartype = spart[2]
    elif filename.startswith("Z"):
        spart = filename.split("_")
        code = spart[3]
        radartype = spart[7]
    else:
        radartype = None
        code = None
    if det_sc:
        radartype = "SC"
    elif det_cd:
        radartype = "CD"
    elif det_cc:
        radartype = "CC"
    if type_assert:
        radartype = type_assert
    if radartype is None:
        raise RadarDecodeError("Radar type undefined")
    return code, radartype


class CinradReader(RadarBase):
    r"""
    Class handling CINRAD radar reading

    Attributes
    ----------
    radartype: str
        type of radar (SA, SB, etc.)
    scantime: datetime.datetime
        time of scan for this data
    code: str
        code for this radar
    angleindex_r: list
        indices of tilts which have reflectivity data
    angleindex_v: list
        indices of tilts which have velocity data
    stationlon: float
        logitude of this radar
    stationlat: float
        latitude of this radar
    radarheight: float
        height of this radar
    name: str
        name of this radar
    Rreso: float
        radial resolution of reflectivity data
    Vreso: float
        radial resolution of velocity data
    a_reso: int
        number of radials in one scan
    el: np.ndarray
        elevation angles for tilts
    drange: float
        current selected radius of data
    elev: float
        elevation angle of current selected tilt
    tilt: int
        current selected tilt
    """

    def __init__(
        self,
        file: Any,
        radar_type: Optional[str] = None,
        file_name: Optional[str] = None,
    ):
        r"""
        Parameters
        ----------
        file: str / obj with read method
            path directed to the file to read / file object
        radar_type: str, optional
            type of radar
        """
        f = prepare_file(file)
        filename = Path(file).name if isinstance(file, str) else ""
        self.code, radartype = _detect_radartype(f, filename, type_assert=radar_type)
        if radar_type:
            if radartype is not radar_type:
                warnings.warn(
                    "Contradictory information from input radar type and\
                              radar type detected from input file."
                )
            self.radartype = radar_type
        else:
            self.radartype = radartype
        f.seek(0)
        if radartype in ["SA", "SB"]:
            self._SAB_handler(f)
        elif radartype in ["CA", "CB"]:
            self._SAB_handler(f, dtype="CAB")
        else:
            try:
                if radartype == "CC":
                    self._CC_handler(f)
                elif radartype in ["SC", "CD"]:
                    self._CD_handler(f)
                else:
                    raise RadarDecodeError("Unrecognized data")
            except Exception as err:
                # Currently there's no good way to differentiate the special
                # SC/CC files, so catch the exception of normal decoding process
                # and try this one if possible
                try:
                    f.seek(0)
                    self._SAB_handler(f, dtype="special")
                except:
                    raise err
        self._update_radar_info()
        f.close()

    def _SAB_handler(self, f: Any, dtype: str = "SAB"):
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

    def _CC_handler(self, f: Any):
        header = np.frombuffer(f.read(1024), CC_header)
        scan_mode = header["ucScanMode"][0]
        if scan_mode < 100:
            raise NotImplementedError("Only VPPI scan mode is supported")
        stop_angle = scan_mode - 100
        self.scantime = (
            datetime.datetime(
                header["ucEYear1"][0] * 100 + header["ucEYear2"][0],
                header["ucEMonth"][0],
                header["ucEDay"][0],
                header["ucEHour"],
                header["ucEMinute"],
                header["ucESecond"],
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

    def _CD_handler(self, f: Any):
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
        if self.radartype in ["SA", "SB", "CA", "CB"]:
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

        Parameters
        ----------
        tilt: int
            index of elevation angle
        drange: float
            radius of data
        dtype: str
            type of product (REF, VEL, etc.)

        Returns
        -------
        ret: ndarray or tuple of ndarray
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
        cut = data.T[: int(np.round(drange / reso))]
        shape_diff = np.round(drange / reso) - cut.shape[0]
        append = np.zeros((int(np.round(shape_diff)), cut.shape[1])) * np.ma.masked
        if dtype in ["VEL", "SW"]:
            try:
                rf = self.data[tilt]["RF"]
            except KeyError:
                pass
            else:
                rf_flag = True
                rf = rf.T[: int(np.round(drange / reso))]
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
        Get radar data

        Parameters
        ----------
        tilt: int
            index of elevation angle
        drange: float
            radius of data
        dtype: str
            type of product (REF, VEL, etc.)

        Returns
        -------
        ds: xarray.Dataset
        """
        task = getattr(self, "task_name", None)
        reso = self.Rreso if dtype == "REF" else self.Vreso
        ret = self.get_raw(tilt, drange, dtype)
        x, y, z, d, a = self.projection(reso)
        shape = ret[0].shape[1] if isinstance(ret, tuple) else ret.shape[1]
        if dtype in ["VEL", "SW"]:
            da = xr.DataArray(ret[0], coords=[a, d], dims=["azimuth", "distance"])
        else:
            da = xr.DataArray(ret, coords=[a, d], dims=["azimuth", "distance"])
        ds = xr.Dataset(
            {dtype: da},
            attrs={
                "elevation": self.elev,
                "range": int(np.round(shape * reso)),
                "scan_time": self.scantime,
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
        if dtype in ["VEL", "SW"]:
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
    Class handling new cinrad standard data reading

    Attributes
    ----------
    scantime: datetime.datetime
        time of scan for this data
    code: str
        code for this radar
    angleindex_r: list
        indices of tilts which have reflectivity data
    angleindex_v: list
        indices of tilts which have velocity data
    stationlon: float
        logitude of this radar
    stationlat: float
        latitude of this radar
    radarheight: float
        height of this radar
    name: str
        name of this radar
    a_reso: int
        number of radials in one scan
    el: np.ndarray
        elevation angles for tilts
    drange: float
        current selected radius of data
    elev: float
        elevation angle of current selected tilt
    tilt: int
        current selected tilt
    """
    # fmt: off
    dtype_corr = {1:'TREF', 2:'REF', 3:'VEL', 4:'SW', 5:'SQI', 6:'CPA', 7:'ZDR', 8:'LDR',
                  9:'RHO', 10:'PHI', 11:'KDP', 12:'CP', 14:'HCL', 15:'CF', 16:'SNRH',
                  17:'SNRV', 32:'Zc', 33:'Vc', 34:'Wc', 35:'ZDRc'}
    # fmt: on
    def __init__(self, file: Any):
        r"""
        Parameters
        ----------
        file: str
            path directed to the file to read
        """
        self.f = prepare_file(file)
        self._parse()
        self.f.close()
        self._update_radar_info()
        # In standard data, station information stored in file
        # has higher priority, so we override some information.
        self.stationlat = self.geo["lat"][0]
        self.stationlon = self.geo["lon"][0]
        self.radarheight = self.geo["height"][0]
        if self.name == "None":
            # Last resort to find info
            self.name = self.code
        self.angleindex_r = self.available_tilt("REF")  # API consistency
        del self.geo

    def _parse(self):
        header = np.frombuffer(self.f.read(32), SDD_header)
        if header["magic_number"] != 0x4D545352:
            raise RadarDecodeError("Invalid standard data")
        site_config = np.frombuffer(self.f.read(128), SDD_site)
        self.code = merge_bytes(site_config["site_code"][0])[:5].decode()
        self.geo = geo = dict()
        geo["lat"] = site_config["Latitude"]
        geo["lon"] = site_config["Longitude"]
        geo["height"] = site_config["ground_height"]
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
            radial_header = np.frombuffer(self.f.read(64), SDD_rad_header)
            if radial_header["zip_type"][0] == 1:  # LZO compression
                raise NotImplementedError("LZO compressed file is not supported")
            self._time_radial.append(
                radial_header["seconds"][0] + radial_header["microseconds"][0]
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

    def get_raw(
        self, tilt: int, drange: Number_T, dtype: str
    ) -> Union[np.ndarray, tuple]:
        r"""
        Get radar raw data

        Parameters
        ----------
        tilt: int
            index of elevation angle
        drange: float
            radius of data
        dtype: str
            type of product (REF, VEL, etc.)

        Returns
        -------
        ret: ndarray or tuple of ndarray
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
        if raw.size == 0:
            warnings.warn("Empty data", RuntimeWarning)
            # Calculate size equivalent
            nrays = len(self.aux[tilt]["azimuth"])
            ngates = int(drange / reso)
            out = np.zeros((nrays, ngates)) * np.ma.masked
            return out
        # Data below 5 are used as reserved codes, which are used to indicate other
        # information instead of real data, so they should be masked.
        data = np.ma.masked_less(raw, 5)
        cut = data[:, : int(drange / reso)]
        shape_diff = np.round(drange / reso) - cut.shape[1]
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
        Get radar data

        Parameters
        ----------
        tilt: int
            index of elevation angle
        drange: float
            radius of data
        dtype: str
            type of product (REF, VEL, etc.)

        Returns
        -------
        ds: xarray.Dataset
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
                    "range": int(shape * reso),
                    "scan_time": self.scantime,
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
                    "scan_time": self.scantime,
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
        r = np.arange(reso, self.drange + reso, reso)
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
