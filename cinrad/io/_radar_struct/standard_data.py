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
    ("max_val", "i4"),
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

l3_hail_table = [
    ("hail_id", "i4"),
    ("hail_azimuth", "f4"),
    ("hail_range", "i4"),
    ("hail_possibility", "i4"),
    ("hail_severe_possibility", "i4"),
    ("hail_size", "f4"),
    ("rcm", "i4"),
]

l3_hail_table_dtype = np.dtype(l3_hail_table)

l3_meso_table = [
    ("feature_id", "i4"),
    ("storm_id", "i4"),
    ("meso_azimuth", "f4"),
    ("meso_range", "i4"),
    ("meso_elevation", "f4"),
    ("meso_avgshr", "f4"),
    ("meso_height", "i4"),
    ("meso_azdia", "i4"),
    ("meso_radius", "i4"),
    ("meso_avgrv", "f4"),
    ("meso_mxrv", "f4"),
    ("meso_top", "i4"),
    ("meso_base", "i4"),
    ("meso_baseazim", "f4"),
    ("meso_baserange", "i4"),
    ("meso_baseelevation", "f4"),
    ("meso_mxtanshr", "f4"),
]

l3_meso_table_dtype = np.dtype(l3_meso_table)

l3_feature_table = [
    ("feature_id", "i4"),
    ("storm_id", "i4"),
    ("feature_type", "i4"),
    ("feature_azimuth", "f4"),
    ("feature_range", "i4"),
    ("feature_elevation", "f4"),
    ("feature_avgshr", "f4"),
    ("feature_height", "i4"),
    ("feature_azdia", "i4"),
    ("feature_radius", "i4"),
    ("feature_avgrv", "f4"),
    ("feature_mxrv", "f4"),
    ("feature_top", "i4"),
    ("feature_base", "i4"),
    ("feature_baseazim", "f4"),
    ("feature_baserange", "i4"),
    ("feature_baseelevation", "f4"),
    ("feature_mxtanshr", "f4"),
]

l3_feature_table_dtype = np.dtype(l3_feature_table)

l3_tvs_table = [
    ("tvs_id", "i4"),
    ("tvs_stormtype", "i4"),
    ("tvs_azimuth", "f4"),
    ("tvs_range", "i4"),
    ("tvs_elevation", "f4"),
    ("tvs_lldv", "f4"),
    ("tvs_avgdv", "f4"),
    ("tvs_mxdv", "f4"),
    ("tvs_mxdvhgt", "i4"),
    ("tvs_depth", "i4"),
    ("tvs_base", "i4"),
    ("tvs_top", "i4"),
    ("tvs_mxshr", "f4"),
    ("tvs_mxshrhgt", "i4"),
]

l3_tvs_table_dtype = np.dtype(l3_tvs_table)

l3_sti_header = [
    ("num_of_storms", "i4"),
    ("num_of_continuous_storms", "i4"),
    ("num_of_components", "i4"),
    ("avg_speed", "f4"),
    ("avg_direction", "f4"),
]

l3_sti_header_dtype = np.dtype(l3_sti_header)

l3_sti_motion = [
    ("azimuth", "f4"),
    ("range", "i4"),
    ("speed", "f4"),
    ("direction", "f4"),
    ("forecast_error", "i4"),
    ("mean_forecast_error", "i4"),
]

l3_sti_motion_dtype = np.dtype(l3_sti_motion)

l3_sti_position = [
    ("azimuth", "f4"),
    ("range", "i4"),
    ("volume_time", "i4"),
]

l3_sti_position_dtype = np.dtype(l3_sti_position)

l3_sti_attribute = [
    ("id", "i4"),
    ("type", "i4"),
    ("num_of_volumes", "i4"),
    ("azimuth", "f4"),
    ("range", "i4"),
    ("height", "i4"),
    ("max_ref", "f4"),
    ("max_ref_height", "i4"),
    ("vil", "f4"),
    ("num_of_components", "i4"),
    ("index_of_first", "i4"),
    ("top_height", "i4"),
    ("index_of_top", "i4"),
    ("bottom_height", "i4"),
    ("index_of_bottom", "i4"),
]

l3_sti_attribute_dtype = np.dtype(l3_sti_attribute)

l3_sti_component = [
    ("height", "i4"),
    ("max_ref", "f4"),
    ("index_of_next", "i4"),
]

l3_sti_component_dtype = np.dtype(l3_sti_component)

l3_sti_adaptation = [
    ("def_dir", "i4"),
    ("def_spd", "f4"),
    ("max_vtime", "i4"),
    ("num_of_past_volume", "i4"),
    ("corr_speed", "f4"),
    ("min_speed", "f4"),
    ("allow_error", "i4"),
    ("frc_intvl", "i4"),
    ("num_frc_intvl", "i4"),
    ("err_intvl", "i4"),
]

l3_sti_adaptation_dtype = np.dtype(l3_sti_adaptation)

l3_vwp_header = [
    ("nyquist_velocity", "f4"),
    ("number_of_vols", "i4"),
    ("wind_speed_max", "f4"),
    ("wind_direction_max", "f4"),
    ("height_max", "f4"),
    ("res", "12c"),
]

l3_vwp_header_dtype = np.dtype(l3_vwp_header)

l3_vwp_table = [
    ("start_time", "i4"),
    ("height", "i4"),
    ("fitvalid", "i4"),
    ("wind_direction", "f4"),
    ("wind_speed", "f4"),
    ("rms_std", "f4"),
    ("res", "8c"),
]

l3_vwp_table_dtype = np.dtype(l3_vwp_table)

l3_swp = [
    ("range", "i4"),
    ("azimuth", "f4"),
    ("swp", "i4"),
]

l3_swp_dtype = np.dtype(l3_swp)

l3_uam = [
    ("range", "i4"),
    ("azimuth", "f4"),
    ("a", "i4"),
    ("b", "i4"),
    ("deg", "i4"),
    ("max1", "f4"),
    ("max2", "f4"),
    ("max3", "f4"),
    ("max4", "f4"),
    ("max5", "f4"),
    ("area", "f4"),
]

l3_uam_dtype = np.dtype(l3_uam)

l3_wer_header =[
    ("elevation", "f4"),
    ("scan_time", "i4"),
    ("center_height", "i4"),
    ("res", "20c"),
]

l3_wer_header_dtype = np.dtype(l3_wer_header)

