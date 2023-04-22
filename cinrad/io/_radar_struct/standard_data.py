# Generated by parse_spec.py
# Do not modify
import numpy as np

generic_header = [
    ("magic_number", "i4"),
    ("major_version", "i2"),
    ("minor_version", "i2"),
    ("generic_type", "i4"),
    ("product_type", "i4"),
    ("res1", "16c"),
]

generic_header_dtype = np.dtype(generic_header)

site_config = [
    ("site_code", "S8"),
    ("site_name", "S32"),
    ("Latitude", "f4"),
    ("Longitude", "f4"),
    ("antenna_height", "i4"),
    ("ground_height", "i4"),
    ("frequency", "f4"),
    ("beam_width_hori", "f4"),
    ("beam_width_vert", "f4"),
    ("RDA_version", "i4"),
    ("radar_type", "i2"),
    ("antenna_gain", "i2"),
    ("trans_loss", "i2"),
    ("recv_loss", "i2"),
    ("other_loss", "i2"),
    ("res2", "46c"),
]

site_config_dtype = np.dtype(site_config)

task_config = [
    ("task_name", "S32"),
    ("task_dsc", "S128"),
    ("polar_type", "i4"),
    ("scan_type", "i4"),
    ("pulse_width", "i4"),
    ("scan_start_time", "i4"),
    ("cut_number", "i4"),
    ("hori_noise", "f4"),
    ("vert_noise", "f4"),
    ("hori_cali", "f4"),
    ("vert_cali", "f4"),
    ("hori_tmp", "f4"),
    ("vert_tmp", "f4"),
    ("ZDR_cali", "f4"),
    ("PHIDP_cali", "f4"),
    ("LDR_cali", "f4"),
    ("res3", "40c"),
]

task_config_dtype = np.dtype(task_config)

cut_config = [
    ("process_mode", "i4"),
    ("wave_form", "i4"),
    ("PRF1", "f4"),
    ("PRF2", "f4"),
    ("dealias_mode", "i4"),
    ("azimuth", "f4"),
    ("elev", "f4"),
    ("start_angle", "f4"),
    ("end_angle", "f4"),
    ("angular_reso", "f4"),
    ("scan_spd", "f4"),
    ("log_reso", "i4"),
    ("dop_reso", "i4"),
    ("max_range1", "i4"),
    ("max_range2", "i4"),
    ("start_range", "i4"),
    ("sample1", "i4"),
    ("sample2", "i4"),
    ("phase_mode", "i4"),
    ("atmos_loss", "f4"),
    ("nyquist_spd", "f4"),
    ("moments_mask", "i8"),
    ("moments_size_mask", "i8"),
    ("misc_filter_mask", "i4"),
    ("SQI_thres", "f4"),
    ("SIG_thres", "f4"),
    ("CSR_thres", "f4"),
    ("LOG_thres", "f4"),
    ("CPA_thres", "f4"),
    ("PMI_thres", "f4"),
    ("DPLOG_thres", "f4"),
    ("res_thres", "4V"),
    ("dBT_mask", "i4"),
    ("dBZ_mask", "i4"),
    ("vel_mask", "i4"),
    ("sw_mask", "i4"),
    ("DP_mask", "i4"),
    ("res_mask", "12V"),
    ("scan_sync", "i4"),
    ("direction", "i4"),
    ("ground_clutter_classifier_type", "i2"),
    ("ground_clutter_filter_type", "i2"),
    ("ground_clutter_filter_notch_width", "i2"),
    ("ground_clutter_filter_window", "i2"),
    ("res4", "72V"),
]

cut_config_dtype = np.dtype(cut_config)

radial_header = [
    ("radial_state", "i4"),
    ("spot_blank", "i4"),
    ("seq_number", "i4"),
    ("radial_number", "i4"),
    ("elevation_number", "i4"),
    ("azimuth", "f4"),
    ("elevation", "f4"),
    ("seconds", "i4"),
    ("microseconds", "i4"),
    ("data_length", "i4"),
    ("moment_number", "i4"),
    ("res5", "i2"),
    ("hori_est_noise", "i2"),
    ("vert_est_noise", "i2"),
    ("zip_type", "c"),
    ("res6", "13c"),
]

radial_header_dtype = np.dtype(radial_header)

moment_header = [
    ("data_type", "i4"),
    ("scale", "i4"),
    ("offset", "i4"),
    ("bin_length", "i2"),
    ("flags", "i2"),
    ("block_length", "i4"),
    ("res", "12c"),
]

moment_header_dtype = np.dtype(moment_header)

product_header = [
    ("product_type", "i4"),
    ("product_name", "S32"),
    ("product_gentime", "i4"),
    ("scan_start_time", "i4"),
    ("data_start_time", "i4"),
    ("data_end_time", "i4"),
    ("proj_type", "i4"),
    ("dtype_1", "i4"),
    ("dtype_2", "i4"),
    ("res", "64c"),
]

product_header_dtype = np.dtype(product_header)

l3_radial_header = [
    ("dtype", "i4"),
    ("scale", "i4"),
    ("offset", "i4"),
    ("bin_length", "i2"),
    ("flags", "i2"),
    ("reso", "i4"),
    ("start_range", "i4"),
    ("max_range", "i4"),
    ("nradial", "i4"),
    ("max_val", "int"),
    ("range_of_max", "i4"),
    ("az_of_max", "f4"),
    ("min_val", "i4"),
    ("range_of_min", "i4"),
    ("az_of_min", "f4"),
    ("res", "8c"),
]

l3_radial_header_dtype = np.dtype(l3_radial_header)

l3_radial_block = [
    ("start_az", "f4"),
    ("angular_reso", "f4"),
    ("nbins", "i4"),
    ("res", "20c"),
]

l3_radial_block_dtype = np.dtype(l3_radial_block)

l3_raster_header = [
    ("dtype", "i4"),
    ("scale", "i4"),
    ("offset", "i4"),
    ("bin_length", "i2"),
    ("flags", "i2"),
    ("row_reso", "i4"),
    ("col_reso", "i4"),
    ("row_side_length", "i4"),
    ("col_side_length", "i4"),
    ("max_val", "i4"),
    ("range_of_max", "i4"),
    ("az_of_max", "f4"),
    ("min_val", "i4"),
    ("range_of_min", "i4"),
    ("az_of_min", "f4"),
    ("res", "8c"),
]

l3_raster_header_dtype = np.dtype(l3_raster_header)

l3_hail_table=[
    ("hail_id", "i4"),
    ("hail_azimuth", "f4"),
    ("hail_range", "i4"),
    ("hail_possibility", "i4"),
    ("hail_severe_possibility", "i4"),
    ("hail_size", "f4"),
    ("rcm", "i4"),
]

l3_hail_table_dtype= np.dtype(l3_hail_table)
