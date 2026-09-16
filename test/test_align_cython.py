from pathlib import Path

import numpy as np

from cinrad.utils import vert_integrated_liquid, echo_top
from cinrad.correct.dealias import dealias_unwrap_2d
from align_cases import VOLUME_CASES, VILD_SKIP, VEL_CASES

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name):
    with np.load(GOLDEN) as data:
        return data[name]


def _finite_max_abs(a, b):
    both_nan = np.isnan(a) & np.isnan(b)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    d = np.abs(np.where(both_nan, 0.0, a) - np.where(both_nan, 0.0, b))
    if d.size == 0:
        return 0.0
    return float(np.max(d))


def _finite_max_rel(a, b):
    both_nan = np.isnan(a) & np.isnan(b)
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.where(scale == 0, 1.0, scale)
    d = np.abs(np.where(both_nan, 0.0, a) - np.where(both_nan, 0.0, b)) / scale
    if d.size == 0:
        return 0.0
    return float(np.max(d))


def _assert_exact(got, want, name):
    if np.array_equal(got, want, equal_nan=True):
        return
    mad = _finite_max_abs(got, want)
    raise AssertionError("%s not bit-identical, max abs diff=%r" % (name, mad))


def _assert_vil(got, want, name):
    if np.array_equal(got, want):
        return
    mad = _finite_max_abs(got, want)
    mrd = _finite_max_rel(got, want)
    if mad <= 1e-12 or mrd <= 1e-12:
        return
    raise AssertionError(
        "%s VIL drift too large: max abs=%r max rel=%r" % (name, mad, mrd)
    )


def test_vil_align_all_volumes():
    for name, factory in VOLUME_CASES.items():
        ref, dist, elev = factory()
        got = vert_integrated_liquid(ref, dist, elev)
        _assert_vil(got, _golden("vil_" + name), "vil_" + name)


def test_vild_align_all_volumes():
    for name, factory in VOLUME_CASES.items():
        if name in VILD_SKIP:
            continue
        ref, dist, elev = factory()
        got = vert_integrated_liquid(ref, dist, elev, density=True)
        _assert_vil(got, _golden("vild_" + name), "vild_" + name)


def test_et_align_all_volumes():
    for name, factory in VOLUME_CASES.items():
        ref, dist, elev = factory()
        _assert_exact(echo_top(ref, dist, elev, 0.0), _golden("et_" + name), "et_" + name)
        _assert_exact(
            echo_top(ref, dist, elev, 137.0), _golden("et_h_" + name), "et_h_" + name
        )


def test_vil_threshold_and_beam_variants():
    ref, dist, elev = VOLUME_CASES["thresh"]()
    _assert_vil(
        vert_integrated_liquid(ref, dist, elev, threshold=40.0),
        _golden("vil_thresh40"),
        "vil_thresh40",
    )
    _assert_exact(
        echo_top(ref, dist, elev, 0.0, threshold=40.0),
        _golden("et_thresh40"),
        "et_thresh40",
    )
    ref, dist, elev = VOLUME_CASES["irregular"]()
    _assert_vil(
        vert_integrated_liquid(ref, dist, elev, threshold=10.0),
        _golden("vil_thresh10"),
        "vil_thresh10",
    )
    _assert_exact(
        echo_top(ref, dist, elev, 0.0, threshold=10.0),
        _golden("et_thresh10"),
        "et_thresh10",
    )
    ref, dist, elev = VOLUME_CASES["mix"]()
    _assert_vil(
        vert_integrated_liquid(ref, dist, elev, beam_width=1.0),
        _golden("vil_bw1"),
        "vil_bw1",
    )


def test_et_site_height_973():
    ref, dist, elev = VOLUME_CASES["et_interp"]()
    _assert_exact(echo_top(ref, dist, elev, 973.0), _golden("et_h973"), "et_h973")


def test_dealias_align_all_fields():
    for name, factory in VEL_CASES.items():
        vdata, nyq = factory()
        _assert_exact(
            dealias_unwrap_2d(vdata, nyq), _golden("dealias_" + name), "dealias_" + name
        )


def test_fortran_order_matches_c_order_golden():
    ref, dist, elev = VOLUME_CASES["hail"]()
    got = vert_integrated_liquid(np.asfortranarray(ref), np.asfortranarray(dist), elev)
    _assert_vil(got, _golden("vil_hail"), "vil_hail_f")
    got_et = echo_top(np.asfortranarray(ref), np.asfortranarray(dist), elev, 0.0)
    _assert_exact(got_et, _golden("et_hail"), "et_hail_f")
    vdata, nyq = VEL_CASES["folded"]()
    got_v = dealias_unwrap_2d(np.asfortranarray(vdata), nyq)
    _assert_exact(got_v, _golden("dealias_folded"), "dealias_folded_f")


def test_float32_integer_like_values():
    # API casts to float64. Integer-like values survive float32 without rounding.
    ref, dist, elev = VOLUME_CASES["small"]()
    got = vert_integrated_liquid(
        ref.astype(np.float32), dist.astype(np.float32), elev.astype(np.float32)
    )
    _assert_vil(got, _golden("vil_small"), "vil_small_f32")
    got_et = echo_top(
        ref.astype(np.float32), dist.astype(np.float32), elev.astype(np.float32), 0.0
    )
    _assert_exact(got_et, _golden("et_small"), "et_small_f32")


def test_vild_single_gate_columns_are_zero():
    ref, dist, elev = VOLUME_CASES["vild_single"]()
    got = vert_integrated_liquid(ref, dist, elev, density=True)
    assert np.array_equal(got, _golden("vild_vild_single"))
    assert np.all(got == 0)


def test_et_highest_tilt_uses_top_height():
    ref, dist, elev = VOLUME_CASES["et_top"]()
    got = echo_top(ref, dist, elev, 0.0)
    _assert_exact(got, _golden("et_et_top"), "et_et_top")
    # columns with top >= 18 should equal height of last tilt, not interpolated
    top_h = dist * np.sin(np.deg2rad(elev[-1])) + dist ** 2 / (2 * 8500)
    use_top = ref[-1] >= 18
    assert np.allclose(got[use_top], top_h[use_top])


def test_vil_gt_vs_et_ge_at_eighteen():
    ref, dist, elev = VOLUME_CASES["exact18"]()
    vil = vert_integrated_liquid(ref, dist, elev)
    et = echo_top(ref, dist, elev, 0.0)
    # VIL uses >; ET uses >=. All gates == 18 so VIL is 0 and ET is nonzero.
    assert np.array_equal(vil, _golden("vil_exact18"))
    assert np.array_equal(et, _golden("et_exact18"), equal_nan=True)
    assert np.all(vil == 0)
    assert np.all(et > 0)
