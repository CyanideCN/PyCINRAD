from io import BytesIO

import pytest

from cinrad.io.level2 import infer_type, CinradReader
from cinrad.error import RadarDecodeError

def test_infer_type_from_fname():
    fill = bytes(200)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, 'Z_RADR_I_Z9200_20000000000000_O_DOR_SA_CAP.bin')
    fake_file.close()
    assert code == 'Z9200'
    assert _type == 'SA'

def test_infer_type_from_file_sc():
    fill = bytes(100) + b"CINRAD/SC" + bytes(50)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, "foo")
    fake_file.close()
    assert code == None
    assert _type == 'SC'

def test_infer_type_from_file_cd():
    fill = bytes(100) + b"CINRAD/CD" + bytes(50)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, "foo")
    fake_file.close()
    assert code == None
    assert _type == 'CD'

def test_infer_type_from_file_cc():
    fill = bytes(116) + b"CINRAD/CC" + bytes(50)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, "foo")
    fake_file.close()
    assert code == None
    assert _type == 'CC'

def test_infer_type_from_incomplete_fname():
    fill = bytes(200)
    fake_file = BytesIO(fill)
    code, _type = infer_type(fake_file, 'Z_RADR_I_Z9200.bin')
    fake_file.close()
    assert code == None
    assert _type == None

def test_missing_radar_type():
    fill = bytes(200)
    fake_file = BytesIO(fill)
    with pytest.raises(RadarDecodeError):
        CinradReader(fake_file)

# --- Tests for StandardData.get_data height filtering ---
import numpy as np
import xarray as xr
from unittest.mock import MagicMock, PropertyMock, patch
import datetime

from cinrad.io.level2 import StandardData # Actual class to test method from

# Helper to create a mock StandardData instance configured for get_data tests
def _create_mock_sd_for_get_data(ref_data_shape, height_values, scan_type="PPI", tilt_idx=0):
    mock_sd = MagicMock(spec=StandardData)

    # Attributes StandardData.get_data relies on
    mock_sd.scan_config = [MagicMock(dop_reso=0.001, log_reso=0.001, nyquist_spd=25.0)] # reso in km for calculations
    mock_sd.code = "TEST"
    mock_sd.name = "TestRadar"
    mock_sd.stationlon = 120.0
    mock_sd.stationlat = 30.0
    mock_sd.scantime = datetime.datetime(2024, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    mock_sd.task_name = "SCAN_TASK"
    mock_sd.scan_type = scan_type
    
    # Mock radarheight (in km for the height() function)
    # Using type() to mock a property on an instance
    type(mock_sd).radarheight = PropertyMock(return_value=0.1) # 100m in km

    # Mock get_raw: returns the primary data array (e.g., 'REF')
    # The values of this array will be checked after filtering.
    # Using a simple np.ones array, but could be more complex.
    raw_data_array = np.ones(ref_data_shape, dtype=float)
    mock_sd.get_raw.return_value = raw_data_array

    if scan_type == "PPI":
        mock_sd.elev = 0.5 # degrees, general elevation for this tilt
        # Mock projection: returns x, y, z (height_values), d, a
        # These are geographical coordinates. height_values are key for the test.
        x_coords = np.zeros(ref_data_shape)
        y_coords = np.zeros(ref_data_shape)
        # distance and azimuth coords, matching shape
        dist_coords = np.arange(ref_data_shape[1], dtype=float) 
        az_coords = np.arange(ref_data_shape[0], dtype=float)
        mock_sd.projection.return_value = (x_coords, y_coords, height_values, dist_coords, az_coords)
    
    elif scan_type == "RHI":
        # For RHI, elevs are per-ray, not a single value for the tilt.
        # self.el[tilt_idx] is used for attributes.
        mock_sd.el = [0.5, 1.5, 2.5] # Example elevation angles for RHI
        
        # aux structure for RHI:
        # aux[tilt_idx]['azimuth'] is a list/array of one azimuth
        # aux[tilt_idx]['elevation'] contains the elevation angles for each ray in this RHI scan.
        # These 'elevation' values are used to compute height (y_cor) in RHI.
        # The shape of height_values for RHI should match ref_data_shape.
        # The test will patch `cinrad.io.level2.height` to control these heights.
        mock_sd.aux = {
            tilt_idx: {
                'azimuth': [30.0], # Fixed azimuth for RHI
                'elevation': np.linspace(0.5, 10.0, ref_data_shape[0]) # Vertical dimension "tilts"
            }
        }
    return mock_sd

def test_standard_data_get_data_ppi_height_filtering():
    ref_shape = (2, 5) # 2 azimuths, 5 gates
    # Heights: one row within (1.0, 2.0), one row outside
    height_values = np.array([[1.0, 1.2, 1.5, 1.8, 1.9],   # Kept by filter (1.0, 2.0)
                              [2.1, 2.2, 2.5, 2.8, 3.0]])  # NaNed by filter (1.0, 2.0)
    
    mock_sd_ppi = _create_mock_sd_for_get_data(ref_shape, height_values, scan_type="PPI")

    tilt_idx, drange_km, dtype_str = 0, 5.0, "REF"

    # Case 1: Filter (1.0, 2.0)
    height_range1 = (1.0, 2.0)
    ds1 = StandardData.get_data(mock_sd_ppi, tilt_idx, drange_km, dtype_str, height_range=height_range1)
    expected_data1 = np.array([[1., 1., 1., 1., 1.],
                               [np.nan, np.nan, np.nan, np.nan, np.nan]])
    np.testing.assert_array_equal(ds1[dtype_str].values, expected_data1)
    assert "height" in ds1.coords # Height coordinate should exist

    # Case 2: No filter (height_range is None)
    ds2 = StandardData.get_data(mock_sd_ppi, tilt_idx, drange_km, dtype_str, height_range=None)
    expected_data2 = np.ones(ref_shape) # Original data from get_raw mock
    np.testing.assert_array_equal(ds2[dtype_str].values, expected_data2)
    assert "height" in ds2.coords

    # Case 3: Filter excludes all data
    height_range3 = (5.0, 6.0) # Assuming all height_values are < 5.0
    ds3 = StandardData.get_data(mock_sd_ppi, tilt_idx, drange_km, dtype_str, height_range=height_range3)
    expected_data3 = np.full(ref_shape, np.nan)
    np.testing.assert_array_equal(ds3[dtype_str].values, expected_data3)

    # Case 4: Filter min_h > max_h
    height_range4 = (3.0, 1.0)
    ds4 = StandardData.get_data(mock_sd_ppi, tilt_idx, drange_km, dtype_str, height_range=height_range4)
    np.testing.assert_array_equal(ds4[dtype_str].values, expected_data3) # Should also be all NaN


@patch('cinrad.io.level2.height') # Patch the imported height function in level2.py
def test_standard_data_get_data_rhi_height_filtering(mock_projection_height):
    ref_shape = (3, 5) # 3 RHI "elevation sweeps", 5 gates
    
    # These are the heights that our patched `cinrad.io.level2.height` will return.
    # This gives us direct control over the `h` variable in `get_data` for RHI.
    controlled_rhi_heights = np.array([
        [0.5, 0.6, 0.7, 0.8, 0.9],  # Row 0: To be NaNed by filter (1.0, 1.5)
        [1.0, 1.1, 1.2, 1.3, 1.4],  # Row 1: To be kept by filter (1.0, 1.5)
        [1.6, 1.7, 1.8, 1.9, 2.0]   # Row 2: To be NaNed by filter (1.0, 1.5)
    ])
    mock_projection_height.return_value = controlled_rhi_heights
    
    mock_sd_rhi = _create_mock_sd_for_get_data(ref_shape, None, scan_type="RHI") # height_values not used by RHI mock setup here

    tilt_idx, drange_km, dtype_str = 0, 5.0, "REF"
    
    # Case 1: Filter (1.0, 1.5) for RHI
    # Note: The filter is inclusive: ds.height >= min_h and ds.height <= max_h
    height_range_rhi1 = (1.0, 1.5) 
    ds_rhi1 = StandardData.get_data(mock_sd_rhi, tilt_idx, drange_km, dtype_str, height_range=height_range_rhi1)

    mock_projection_height.assert_called_once() # Ensure our patched height function was used

    # Original data is np.ones(ref_shape) due to _create_mock_sd_for_get_data -> get_raw
    expected_data_rhi1 = np.full(ref_shape, np.nan)
    expected_data_rhi1[1, :] = 1.0 # Middle row (index 1) corresponds to heights 1.0-1.4
    
    np.testing.assert_array_equal(ds_rhi1[dtype_str].values, expected_data_rhi1)
    assert "height" in ds_rhi1.coords # ds.height should be populated from y_cor
    assert "y_cor" in ds_rhi1.coords # y_cor should also be there for RHI

    # Case 2: RHI with no filter
    # Need to reset mock call count if checking per call, or just let it be.
    # For simplicity, we are not checking call count for the no-filter case here.
    # The important part is that the data is not filtered.
    mock_projection_height.return_value = controlled_rhi_heights # Ensure it's set for this call too
    ds_rhi2 = StandardData.get_data(mock_sd_rhi, tilt_idx, drange_km, dtype_str, height_range=None)
    expected_data_rhi2 = np.ones(ref_shape)
    np.testing.assert_array_equal(ds_rhi2[dtype_str].values, expected_data_rhi2)
    assert "height" in ds_rhi2.coords
    assert ds_rhi2.height.shape == ref_shape