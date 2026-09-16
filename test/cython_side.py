# Load original Cython extensions from a side path (not part of this package).
import importlib.util
import os
from pathlib import Path

import numpy as np

DEFAULT_SIDE = Path(os.environ.get("PYCINRAD_CYTHON_SIDE", "/tmp/pycy-old"))


def _load_so(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s from %s" % (modname, path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_so(root, rel_glob):
    root = Path(root)
    hits = list(root.glob(rel_glob))
    if not hits:
        raise FileNotFoundError("no %s under %s" % (rel_glob, root))
    return hits[0]


def load_cython_side(root=DEFAULT_SIDE):
    utils = _load_so(
        "cinrad._utils",
        find_so(root, "cinrad/_utils.cpython-*.so"),
    )
    unwrap = _load_so(
        "cinrad.correct._unwrap_2d",
        find_so(root, "cinrad/correct/_unwrap_2d.cpython-*.so"),
    )
    return utils, unwrap


def cython_dealias_unwrap_2d(unwrap_mod, vdata, nyquist_vel):
    scaled_sweep = np.array(vdata * np.pi / nyquist_vel, copy=True, dtype=np.float64)
    sweep_mask = np.isnan(vdata)
    scaled_sweep[sweep_mask] = 0
    wrapped = np.require(scaled_sweep, np.float64, ["C"])
    mask = np.require(sweep_mask, np.uint8, ["C"])
    unwrapped = np.empty_like(wrapped, dtype=np.float64, order="C")
    unwrap_mod.unwrap_2d(wrapped, mask, unwrapped, [True, False])
    return unwrapped * nyquist_vel / np.pi
