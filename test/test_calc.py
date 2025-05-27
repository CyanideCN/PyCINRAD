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

# --- Tests for quick_cr ---
import xarray as xr
from cinrad.calc import quick_cr
from cinrad.grid import grid_2d # Used for reference calculation

def _create_sample_rlist_for_quick_cr(num_tilts: int, common_shape=(3, 4), resolution=(1.0, 1.0)):
    """
    Creates a list of xr.Dataset objects mimicking radar tilts for quick_cr testing.
    """
    r_list = []
    
    # Define common longitude and latitude grids for simplicity
    # These would typically be 2D arrays from radar projection
    # For grid_2d, input lon/lat are N_az x N_gates.
    # Let's assume polar data that will be gridded.
    # For testing quick_cr, the actual projection details are less critical than
    # having REF, longitude, latitude fields that grid_2d can process.

    # Mocking values as if they came from radar projection (polar -> cartesian-like for grid_2d)
    # grid_2d expects longitude and latitude to be 2D (azimuth, distance)
    n_azimuths = 10
    n_gates = 5
    
    base_lon = np.linspace(120, 120.1, n_gates)
    base_lat = np.linspace(30, 30.1, n_gates)

    # Create common target grid for simpler reference calculation
    # This is what x_out, y_out in grid_2d would be based on resolution
    # For simplicity, let's make the gridded output shape `common_shape`

    for i in range(num_tilts):
        # Create slightly different REF data for each tilt
        ref_data = np.full((n_azimuths, n_gates), 10.0 * (i + 1), dtype=float)
        if i % 2 == 0:
            ref_data[i % n_azimuths, i % n_gates] = np.nan # Introduce some NaNs
            ref_data[(i+1) % n_azimuths, (i+1) % n_gates] = 50.0 + i # Some varying values
        else:
            ref_data[i % n_azimuths, i % n_gates] = 5.0 * (i + 1)
            ref_data[(i+1) % n_azimuths, (i+1) % n_gates] = np.nan


        # Mock longitude/latitude data (could be more realistic if needed)
        # These should be 2D arrays [n_azimuths, n_gates]
        lon_vals = np.array([base_lon + 0.01 * i] * n_azimuths)
        lat_vals = np.array([base_lat + 0.01 * i] * n_azimuths)

        ds_attrs = {
            "scan_time": f"2024-01-01T0{i}:00:00Z",
            "site_name": f"TestSite_Tilt{i}",
            "unique_attr_tilt0": "present" if i == 0 else "absent"
        }
        
        tilt_ds = xr.Dataset(
            data_vars={
                "REF": (("azimuth", "distance"), ref_data),
                "longitude": (("azimuth", "distance"), lon_vals),
                "latitude": (("azimuth", "distance"), lat_vals),
            },
            coords={
                "azimuth": np.linspace(0, 350, n_azimuths),
                "distance": np.linspace(1, n_gates, n_gates)
            },
            attrs=ds_attrs
        )
        r_list.append(tilt_ds)
    return r_list

def test_quick_cr_numerical_consistency_and_attrs():
    """
    Tests quick_cr for numerical consistency with a reference calculation (np.nanmax)
    and checks attribute handling.
    """
    resolution = (0.05, 0.05) # Resolution for gridding, degrees for lat/lon

    # Test with multiple tilts
    r_list_multi_tilt = _create_sample_rlist_for_quick_cr(num_tilts=3, resolution=resolution)
    
    # --- Reference Calculation ---
    gridded_tilts_ref = []
    x_grid_ref, y_grid_ref = None, None

    # First tilt to establish the grid
    r0, x0, y0 = grid_2d(
        r_list_multi_tilt[0]["REF"].values,
        r_list_multi_tilt[0]["longitude"].values,
        r_list_multi_tilt[0]["latitude"].values,
        resolution=resolution
    )
    gridded_tilts_ref.append(r0)
    x_grid_ref, y_grid_ref = x0, y0
    
    for tilt_ds in r_list_multi_tilt[1:]:
        r_gridded, _, _ = grid_2d(
            tilt_ds["REF"].values,
            tilt_ds["longitude"].values,
            tilt_ds["latitude"].values,
            x_out=x_grid_ref,
            y_out=y_grid_ref,
            resolution=resolution
        )
        gridded_tilts_ref.append(r_gridded)
    
    reference_cr_data = np.nanmax(np.stack(gridded_tilts_ref, axis=0), axis=0)

    # --- Current quick_cr Calculation ---
    result_ds_multi = quick_cr(r_list_multi_tilt, resolution=resolution)

    # --- Assertions for multiple tilts ---
    assert "CR" in result_ds_multi
    np.testing.assert_allclose(
        result_ds_multi["CR"].data, reference_cr_data, nan_equal=True,
        err_msg="quick_cr output differs from np.nanmax reference for multiple tilts"
    )
    assert result_ds_multi.attrs["elevation"] == 0
    # Check if attributes are from the first tilt (r_list[0])
    assert result_ds_multi.attrs["site_name"] == r_list_multi_tilt[0].attrs["site_name"]
    assert result_ds_multi.attrs["unique_attr_tilt0"] == "present"


    # Test with a single tilt (covers the attribute bug fix)
    r_list_single_tilt = _create_sample_rlist_for_quick_cr(num_tilts=1, resolution=resolution)
    
    # Reference for single tilt is just the gridded first tilt
    reference_cr_single_data, _, _ = grid_2d(
        r_list_single_tilt[0]["REF"].values,
        r_list_single_tilt[0]["longitude"].values,
        r_list_single_tilt[0]["latitude"].values,
        resolution=resolution
    )

    result_ds_single = quick_cr(r_list_single_tilt, resolution=resolution)
    
    # --- Assertions for single tilt ---
    assert "CR" in result_ds_single
    np.testing.assert_allclose(
        result_ds_single["CR"].data, reference_cr_single_data, nan_equal=True,
        err_msg="quick_cr output differs from gridded single tilt for single tilt input"
    )
    assert result_ds_single.attrs["elevation"] == 0
    assert result_ds_single.attrs["site_name"] == r_list_single_tilt[0].attrs["site_name"]
    assert result_ds_single.attrs["unique_attr_tilt0"] == "present"

    # Test with empty r_list (should raise ValueError due to @require or internal check)
    # Note: @require decorator should catch this. If not, the internal check in quick_cr will.
    # This test might be more for the decorator or initial checks than quick_cr's core logic.
    # import pytest # Add to top imports if not there
    # with pytest.raises(ValueError):
    #    quick_cr([], resolution=resolution) 
    # This test for empty list is commented out as @require handles it, and this test
    # focuses on numerical consistency of the main algorithm and attribute handling.
    # If @require was not present, the explicit check `if not r_list:` in quick_cr would be tested.
