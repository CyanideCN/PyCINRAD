import numpy as np
import pytest
from xarray import Dataset

from cinrad.calc import hydro_class


@pytest.mark.parametrize(
    "band, expected", [("S", [5, 9, 4]), ("C", [5, 9, 4]), ("X", [2, 9, 4])]
)
def test_hydro_class_preserves_polarimetric_variable_order(band, expected):
    # Nonzero, distinct KDP (deg/km) and RHO (dimensionless).
    # Labels from scalar evaluation of the shipped fuzzy membership model:
    # beta_Z * (0.8*beta_ZDR + beta_KDP + 0.8*beta_RHO) / 2.6.
    dims = ["azimuth", "distance"]
    z = Dataset({"REF": (dims, [[30.0, 55.0, 10.0, np.nan]])})
    zdr = Dataset({"ZDR": (dims, [[1.0, 0.0, 0.2, 1.0]])})
    rho = Dataset({"RHO": (dims, [[0.98, 0.9, 0.99, 0.98]])})
    kdp = Dataset({"KDP": (dims, [[0.3, 1.0, 0.0, 0.3]])})

    result = hydro_class(z, zdr, rho, kdp, band=band)

    np.testing.assert_array_equal(result.cHCL.values, [expected + [np.nan]])
    assert result.cHCL.dims == tuple(dims)
    assert "REF" not in result
    assert "REF" in z
