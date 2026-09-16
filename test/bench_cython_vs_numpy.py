# Local wall-time comparison of original Cython vs numpy/skimage. Not a pytest.
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from align_cases import volume_sa230, vel_sa460
from cython_side import load_cython_side, cython_dealias_unwrap_2d
from real_sample import EXTRACTED, find_sample_file, extract_volume, extract_vel

from cinrad.utils import vert_integrated_liquid, echo_top
from cinrad.correct.dealias import dealias_unwrap_2d


def _median(fn, n=7, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), times


def _load_real():
    if EXTRACTED.is_file():
        with np.load(EXTRACTED) as z:
            ref = np.ascontiguousarray(z["ref"], dtype=np.float64)
            dist = np.ascontiguousarray(z["dist"], dtype=np.float64)
            elev = np.ascontiguousarray(z["elev"], dtype=np.float64)
            vel = np.ascontiguousarray(z["vel_t1"], dtype=np.float64)
            nyq = float(z["nyq_t1"])
        return "z9250-extracted", ref, dist, elev, vel, nyq
    sample = find_sample_file()
    if sample is None:
        return None
    vol = extract_volume(sample)
    vel, nyq = extract_vel(vol["reader"], 1)
    return "z9250-io", vol["ref"], vol["dist"], vol["elev"], vel, nyq


def _row(label, old_fn, new_fn):
    old_med, _ = _median(old_fn)
    new_med, _ = _median(new_fn)
    ratio = new_med / old_med if old_med > 0 else float("inf")
    print(
        "%-28s %12.3f %12.3f %8.2fx"
        % (label, old_med * 1000, new_med * 1000, ratio)
    )
    return label, old_med, new_med, ratio


def _jobs(utils, unwrap, ref, dist, elev, vel, nyq):
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    dist = np.ascontiguousarray(dist, dtype=np.float64)
    elev = np.ascontiguousarray(elev, dtype=np.float64)
    vel = np.ascontiguousarray(vel, dtype=np.float64)
    return [
        (
            "vert_integrated_liquid",
            lambda: utils.vert_integrated_liquid(ref, dist, elev),
            lambda: vert_integrated_liquid(ref, dist, elev),
        ),
        (
            "echo_top",
            lambda: utils.echo_top(ref, dist, elev, 0.0),
            lambda: echo_top(ref, dist, elev, 0.0),
        ),
        (
            "dealias_unwrap_2d",
            lambda: cython_dealias_unwrap_2d(unwrap, vel, nyq),
            lambda: dealias_unwrap_2d(vel, nyq),
        ),
    ]


def main():
    utils, unwrap = load_cython_side()
    rows = []
    cases = []
    real = _load_real()
    if real is not None:
        src, ref, dist, elev, vel, nyq = real
        cases.append((src, ref, dist, elev, vel, nyq))
    sref, sdist, selev = volume_sa230()
    svel, snyq = vel_sa460()
    cases.append(("sa230+sa460-synthetic", sref, sdist, selev, svel, snyq))
    print("%-28s %12s %12s %8s" % ("fn", "cython_ms", "numpy_ms", "ratio"))
    for src, ref, dist, elev, vel, nyq in cases:
        print("input", src, "ref", np.shape(ref), "vel", np.shape(vel))
        for label, old_fn, new_fn in _jobs(utils, unwrap, ref, dist, elev, vel, nyq):
            rows.append((src,) + _row("%s/%s" % (src, label), old_fn, new_fn))
    return rows


if __name__ == "__main__":
    main()
