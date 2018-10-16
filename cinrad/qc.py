# -*- coding: utf-8 -*-
# Author: Du puyuan

from .grid import resample
from .constants import deg2rad

import numpy as np
from scipy import interpolate

def gate_filter(v, r, w, power_filter=False):
    if v.include_rf:
        data = v.data[0]
    else:
        data = v.data
    r2 = list()
    for i in r.data:
        f = interpolate.interp1d([0] + r.dist.tolist(), [0] + i.tolist())
        r2.append(f(v.dist))
    r2 = np.asarray(r2)
    r_ = resample(r2, v.dist, r.az, v.reso, data.shape[0] - 1)
    #filter W>2 when R<16
    r_mask = r_[0] < 16
    w_mask = w.data > 2
    step1 = np.logical_and(r_mask, w_mask)
    #filter sporadic noise
    if power_filter:
        w_mask2 = w.data > 1.5
        power = np.ones(v.az.shape)[:, None] * v.dist * 0.25
        r_mask2 = r_[0] < power
        step2 = np.logical_and(r_mask2, w_mask2)
        return np.logical_or(step1, step2)
    else:
        return step1