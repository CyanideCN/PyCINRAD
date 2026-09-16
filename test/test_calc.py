from pathlib import Path

import numpy as np
from cinrad.utils import vert_integrated_liquid, vert_integrated_liquid_py, echo_top, echo_top_py
from align_cases import volume_small, volume_mid

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name):
    with np.load(GOLDEN) as data:
        return data[name]


def test_vil():
    a, b, c = volume_small()
    vil = vert_integrated_liquid(a, b, c)
    true_vil = np.array(
        [
            [0.0, 0.00141174, 0.00322052],
            [0.0, 0.00209499, 0.0047792],
            [0.0, 0.00310893, 0.00709224],
        ]
    )
    assert np.allclose(vil, true_vil)


def test_vil_cy2py():
    a, b, c = volume_mid()
    vil = vert_integrated_liquid(a, b, c)
    vil2 = vert_integrated_liquid_py(a, b, c)
    assert np.allclose(vil, vil2)


def test_et():
    a, b, c = volume_small()
    et = echo_top(a, b, c, 0)
    true_et = np.array(
        [
            [0.0, 0.03495832023191273, 0.070034287522649],
            [0.0, 0.03495832023191273, 0.070034287522649],
            [0.0, 0.03495832023191273, 0.070034287522649],
        ]
    )
    assert np.array_equal(et, true_et)


def test_et_cy2py():
    a, b, c = volume_mid()
    et = echo_top(a, b, c, 0)
    et2 = echo_top_py(a, b, c, 0)
    assert np.array_equal(et, et2)


def test_no_cython_extension():
    import importlib.util

    assert importlib.util.find_spec("cinrad._utils") is None
    assert importlib.util.find_spec("cinrad.correct._unwrap_2d") is None
