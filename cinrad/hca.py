# -*- coding: utf-8 -*-
# Author: Puyuan Du
# Rewrite from https://github.com/YvZheng/pycwr/blob/master/pycwr/retrieve/HID.py

import os

import numpy as np

from cinrad.constants import MODULE_DIR
from cinrad._typing import Boardcast_T

PARAMS = np.load(os.path.join(MODULE_DIR, "data", "hca_params.npy"))

BAND_MAPPING = {"S": 0, "C": 1, "X": 2}
DEFAULT_WEIGHTS = [0.8, 1.0, 0.8]
# ZDR KDP RHO


def beta_func(
    x: Boardcast_T, m: Boardcast_T, a: Boardcast_T, b: Boardcast_T
) -> Boardcast_T:
    return 1 / (1 + ((x - m) / a) ** (2 * b))


def hydro_class(
    z: Boardcast_T,
    zdr: Boardcast_T,
    kdp: Boardcast_T,
    rho: Boardcast_T,
    band: str = "S",
):
    r"""
    Types:           Species #:
    -------------------------------
    Drizzle                  1

    Rain                     2

    Ice Crystals             3

    Dry Aggregates Snow      4

    Wet Snow                 5

    Vertical Ice             6

    Low-Density Graupel      7

    High-Density Graupel     8

    Hail                     9

    Big Drops                10
    """
    params = PARAMS[BAND_MAPPING[band]]
    # Process variables
    z = np.repeat(z[:, None], 10, axis=1)
    zdr = np.repeat(zdr[:, None], 10, axis=1)
    kdp = np.repeat(kdp[:, None], 10, axis=1)
    rho = np.repeat(rho[:, None], 10, axis=1)
    p_z = beta_func(z, params[0][:, 0], params[0][:, 1], params[0][:, 2])
    p_zdr = beta_func(zdr, params[1][:, 0], params[1][:, 1], params[1][:, 2])
    p_kdp = beta_func(kdp, params[2][:, 0], params[2][:, 1], params[2][:, 2])
    p_rho = beta_func(rho, params[3][:, 0], params[3][:, 1], params[3][:, 2])
    hclass = np.average([p_zdr, p_kdp, p_rho], weights=DEFAULT_WEIGHTS, axis=0)
    hclass *= p_z
    return np.argmax(hclass, axis=1) + 1
