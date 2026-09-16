import numpy as np
import pytest

from cinrad.utils import vert_integrated_liquid, echo_top
from cinrad.correct.dealias import dealias_unwrap_2d
from real_sample import (
    SAMPLE_NAME,
    EXTRACTED,
    cinrad_data_radar_files,
    ensure_sample_file,
    extract_volume,
    extract_vel,
)
from test_align_cython import _assert_vil, _assert_exact, _golden


def test_cinrad_data_contains_no_level2_or_product_files():
    root, found = cinrad_data_radar_files()
    # cinrad_data v0.1 only ships shapefile/ and font/ (see package layout).
    assert root.is_dir()
    assert (root / "shapefile").is_dir()
    assert (root / "font").is_dir()
    assert found == []


def test_real_z9250_extracted_arrays_match_cython():
    assert EXTRACTED.is_file(), "missing %s; run test/freeze_cython_goldens.py" % EXTRACTED
    with np.load(EXTRACTED) as z:
        ref = z["ref"]
        dist = z["dist"]
        elev = z["elev"]
        h = float(z["radarheight"])
        vel1, nyq1 = z["vel_t1"], float(z["nyq_t1"])
        vel5, nyq5 = z["vel_t5"], float(z["nyq_t5"])
        assert np.array(z["filename"]).item() == SAMPLE_NAME
    _assert_vil(vert_integrated_liquid(ref, dist, elev), _golden("vil_z9250"), "vil_z9250")
    _assert_vil(
        vert_integrated_liquid(ref, dist, elev, density=True),
        _golden("vild_z9250"),
        "vild_z9250",
    )
    _assert_exact(echo_top(ref, dist, elev, h), _golden("et_z9250"), "et_z9250")
    _assert_exact(
        dealias_unwrap_2d(vel1, nyq1), _golden("dealias_z9250_t1"), "dealias_z9250_t1"
    )
    _assert_exact(
        dealias_unwrap_2d(vel5, nyq5), _golden("dealias_z9250_t5"), "dealias_z9250_t5"
    )


def test_real_z9250_via_cinrad_io_matches_extracted_and_cython():
    try:
        path = ensure_sample_file()
    except Exception as exc:
        pytest.skip("public SA sample not available: %s" % exc)
    vol = extract_volume(path)
    with np.load(EXTRACTED) as z:
        assert np.array_equal(vol["ref"], z["ref"], equal_nan=True)
        assert np.array_equal(vol["dist"], z["dist"])
        assert np.array_equal(vol["elev"], z["elev"])
    ref, dist, elev = vol["ref"], vol["dist"], vol["elev"]
    _assert_vil(vert_integrated_liquid(ref, dist, elev), _golden("vil_z9250"), "vil_z9250_io")
    _assert_vil(
        vert_integrated_liquid(ref, dist, elev, density=True),
        _golden("vild_z9250"),
        "vild_z9250_io",
    )
    _assert_exact(
        echo_top(ref, dist, elev, vol["radarheight"]), _golden("et_z9250"), "et_z9250_io"
    )
    v1, n1 = extract_vel(vol["reader"], 1)
    v5, n5 = extract_vel(vol["reader"], 5)
    _assert_exact(dealias_unwrap_2d(v1, n1), _golden("dealias_z9250_t1"), "dealias_z9250_t1_io")
    _assert_exact(dealias_unwrap_2d(v5, n5), _golden("dealias_z9250_t5"), "dealias_z9250_t5_io")
