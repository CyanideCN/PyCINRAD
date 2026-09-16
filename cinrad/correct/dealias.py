# -*- coding: utf-8 -*-
# Author: PyCINRAD Developers

import numpy as np
from xarray import Dataset
from skimage.restoration import unwrap_phase


def dealias_unwrap_2d(vdata: np.ndarray, nyquist_vel: float) -> np.ndarray:
    """Dealias using 2D phase unwrapping (sweep-by-sweep)."""
    scaled_sweep = vdata * np.pi / nyquist_vel
    sweep_mask = np.isnan(vdata)
    scaled_sweep[sweep_mask] = 0
    wrapped = np.ma.array(scaled_sweep, mask=sweep_mask)
    unwrapped = unwrap_phase(wrapped, wrap_around=(True, False))
    return np.asarray(unwrapped) * nyquist_vel / np.pi


def dealias(v_data: Dataset) -> Dataset:
    v_field = v_data["VEL"].data
    nyq = v_data.attrs.get("nyquist_vel")
    out_data = dealias_unwrap_2d(v_field, nyq)
    out_masked = np.ma.array(out_data, mask=np.isnan(v_field))
    v_ret = v_data.copy()
    v_ret["VEL"] = (tuple(v_data.dims.keys()), out_masked)
    return v_ret
