import numpy as np
from cinrad.utils import vert_integrated_liquid, vert_integrated_liquid_py, echo_top, echo_top_py


def test_vil():
    a = np.arange(0, 27, 1, dtype=np.double).reshape(3, 3, 3)
    b = np.broadcast_to(np.arange(0, 3), (3, 3))
    b = np.ascontiguousarray(b, dtype=np.double)
    c = np.arange(0, 3, 1, dtype=np.double)

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
    a = np.arange(0, 64, 1, dtype=np.double).reshape(4, 4, 4)
    b = np.broadcast_to(np.arange(0, 4), (4, 4))
    b = np.ascontiguousarray(b, dtype=np.double)
    c = np.arange(0, 4, 1, dtype=np.double)

    vil = vert_integrated_liquid(a, b, c)
    vil2 = vert_integrated_liquid_py(a, b, c)
    assert np.allclose(vil, vil2)

def test_et():
    a = np.arange(0, 27, 1, dtype=np.double).reshape(3, 3, 3)
    b = np.broadcast_to(np.arange(0, 3), (3, 3))
    b = np.ascontiguousarray(b, dtype=np.double)
    c = np.arange(0, 3, 1, dtype=np.double)

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
    a = np.arange(0, 64, 1, dtype=np.double).reshape(4, 4, 4)
    b = np.broadcast_to(np.arange(0, 4), (4, 4))
    b = np.ascontiguousarray(b, dtype=np.double)
    c = np.arange(0, 4, 1, dtype=np.double)

    et = echo_top(a, b, c, 0)
    et2 = echo_top_py(a, b, c, 0)

    assert np.array_equal(et, et2)
