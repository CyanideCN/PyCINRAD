# -*- coding: utf-8 -*-
# Author: Puyuan Du

import numpy as np
from xarray import Dataset

try:
    from cinrad.correct._unwrap_2d import unwrap_2d
except ImportError:
    from cinrad.error import RadarCalculationError, ExceptionOnCall

    unwrap_2d = ExceptionOnCall(
        RadarCalculationError,
        "Cython is not installed, velocity dealias function cannot be used.",
    )


def dealias_unwrap_2d(vdata: np.ndarray, nyquist_vel: float) -> np.ndarray:
    """ Dealias using 2D phase unwrapping (sweep-by-sweep). """
    scaled_sweep = vdata * np.pi / nyquist_vel
    sweep_mask = np.isnan(vdata)
    scaled_sweep[sweep_mask] = 0
    wrapped = np.require(scaled_sweep, np.float64, ["C"])
    mask = np.require(sweep_mask, np.uint8, ["C"])
    unwrapped = np.empty_like(wrapped, dtype=np.float64, order="C")
    unwrap_2d(wrapped, mask, unwrapped, [True, False])
    return unwrapped * nyquist_vel / np.pi


def dealias(v_data: Dataset) -> Dataset:
    v_field = v_data["VEL"].data
    nyq = v_data.attrs.get("nyquist_vel")
    out_data = dealias_unwrap_2d(v_field, nyq)
    out_masked = np.ma.array(out_data, mask=np.isnan(v_field))
    v_ret = v_data.copy()
    v_ret["VEL"] = (["azimuth", "distance"], out_masked)
    return v_ret
