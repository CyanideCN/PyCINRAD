# -*- coding: utf-8 -*-
# Author: Puyuan Du

import numpy as np

# fmt: off
__all__ = [
    'CC_param', 'CC_header', 'CC_data', 'SDD_header', 'SDD_site', 'SDD_task',
    'SDD_cut', 'SDD_rad_header', 'SDD_mom_header', 'SAB_dtype', 'CAB_dtype',
    'SWAN_dtype', 'CD_dtype', 'CD_DATA', 'SDD_pheader', 'L3_radial', 'L3_rblock',
    'S_SPECIAL_dtype', 'CC2_header', 'CC2_obs', 'CC2_data', 'CC2_other', 'PA_radial',
    'L3_raster','L3_hail','L3_meso','L3_feature','L3_tvs','L3_sti_header','L3_sti_motion',
    'L3_sti_position','L3_sti_attribute','L3_sti_component','L3_sti_adaptation',
    'L3_vwp_header','L3_vwp','L3_swp','L3_uam','mocm_dtype'
]
# fmt: on
from cinrad.io._radar_struct.CC import (
    scan_param_dtype as CC_param,
    header_dtype as CC_header,
    data_dtype as CC_data,
)
from cinrad.io._radar_struct.standard_data import (
    generic_header_dtype as SDD_header,
    site_config_dtype as SDD_site,
    task_config_dtype as SDD_task,
    cut_config_dtype as SDD_cut,
    radial_header_dtype as SDD_rad_header,
    moment_header_dtype as SDD_mom_header,
    product_header_dtype as SDD_pheader,
    l3_radial_header_dtype as L3_radial,
    l3_radial_block_dtype as L3_rblock,
    l3_raster_header_dtype as L3_raster,
    l3_hail_table_dtype as L3_hail,
    l3_meso_table_dtype as L3_meso,
    l3_feature_table_dtype as L3_feature,
    l3_tvs_table_dtype as L3_tvs,
    l3_sti_header_dtype as L3_sti_header,
    l3_sti_motion_dtype as L3_sti_motion,
    l3_sti_position_dtype as L3_sti_position,
    l3_sti_attribute_dtype as L3_sti_attribute,
    l3_sti_component_dtype as L3_sti_component,
    l3_sti_adaptation_dtype as L3_sti_adaptation,
    l3_vwp_header_dtype as L3_vwp_header,
    l3_vwp_table_dtype as L3_vwp,
    l3_swp as L3_swp,
    l3_uam_dtype as L3_uam,
)
from cinrad.io._radar_struct.CD import (
    radarsite_dtype as CD_site,
    performance_dtype as CD_perf,
    observation_param_dtype as CD_obs,
)
from cinrad.io._radar_struct.CC2 import (
    header_dtype as CC2_header,
    observation_param_dtype as CC2_obs,
    other_info_dtype as CC2_other,
    data_block_dtype as CC2_data,
)
from cinrad.io._radar_struct.PA import radial_dtype as PA_radial

_S_HEADER = [("spare", "u2", 7), ("a", "u2"), ("res", "u2", 6)]

_S_INFO = [
    ("time", "u4"),
    ("day", "u2"),
    ("unambiguous_distance", "u2"),
    ("azimuth", "u2"),
    ("radial_num", "u2"),
    ("radial_state", "u2"),
    ("elevation", "u2"),
    ("el_num", "u2"),
    ("first_gate_r", "u2"),
    ("first_gate_v", "u2"),
    ("gate_length_r", "u2"),
    ("gate_length_v", "u2"),
    ("gate_num_r", "u2"),
    ("gate_num_v", "u2"),
    ("sector_num", "u2"),
    ("system_coff", "u4"),
    ("r_pointer", "u2"),
    ("v_pointer", "u2"),
    ("w_pointer", "u2"),
    ("v_reso", "u2"),
    ("vcp_mode", "u2"),
    ("res2", "u2", 4),
    ("r_pointer_2", "u2"),
    ("v_pointer_2", "u2"),
    ("w_pointer_2", "u2"),
    ("nyquist_vel", "u2"),
]

_S_RES = [("res3", "u2", 19)]
_S_RES_2 = [("res3", "u2", 21)]

_SAB_DATA = [("r", "u1", 460), ("v", "u1", 920), ("w", "u1", 920), ("res4", "u2", 2)]

_CAB_DATA = [("r", "u1", 800), ("v", "u1", 1600), ("w", "u1", 1600), ("res4", "u2", 2)]

_SPECIAL_DATA = [("r", "u1", 1000), ("v", "u1", 1000), ("w", "u1", 1000)]

SAB_dtype = np.dtype(_S_HEADER + _S_INFO + _S_RES + _SAB_DATA)
CAB_dtype = np.dtype(_S_HEADER + _S_INFO + _S_RES + _CAB_DATA)
S_SPECIAL_dtype = np.dtype(_S_HEADER + _S_INFO + _S_RES_2 + _SPECIAL_DATA)

SWAN_HEADER = [
    ("data_type", "12c"),
    ("data_name", "38c"),
    ("name", "8c"),
    ("version", "8c"),
    ("year", "u2"),
    ("month", "u2"),
    ("day", "u2"),
    ("hour", "u2"),
    ("minute", "u2"),
    ("interval", "u2"),
    ("x_grid_num", "u2"),
    ("y_grid_num", "u2"),
    ("z_grid_num", "u2"),
    ("radar_count", "i4"),
    ("start_lon", "f4"),
    ("start_lat", "f4"),
    ("center_lon", "f4"),
    ("center_lat", "f4"),
    ("x_reso", "f4"),
    ("y_reso", "f4"),
    ("height", "40f4"),
    ("station_names", "20S16"),
    ("station_lons", "20f4"),
    ("station_lats", "20f4"),
    ("station_alts", "20f4"),
    ("mosaic_flags", "20B"),
    ("m_data_type", "i2"),
    ("dimension", "i2"),
    ("res", "168c"),
]

SWAN_dtype = np.dtype(SWAN_HEADER)

CD_dtype = np.dtype(
    [("site_info", CD_site), ("performance", CD_perf), ("obs", CD_obs), ("res", "163c")]
)

_CD_record = np.dtype([("m_dbz", "B"), ("m_vel", "B"), ("m_undbz", "B"), ("m_sw", "B")])

CD_DATA = np.dtype(
    [
        ("s_az", "u2"),
        ("s_el", "u2"),
        ("e_az", "u2"),
        ("e_el", "u2"),
        ("rec", _CD_record, 998),
    ]
)

MOCM_HEADER = [
    ("label", "4c"),
    ("version", "4c"),
    ("file_bytes", "i4"),
    ("mosaic_id", "u2"),
    ("coordinate","u2"),
    ("varname","8c"),
    ("description", "64c"),
    ("block_pos","i4"),
    ("block_len","i4"),
    ("time_zone","i4"),
    ("year", "u2"),
    ("month", "u2"),
    ("day", "u2"),
    ("hour", "u2"),
    ("minute", "u2"),
    ("second", "u2"),
    ("obs_seconds", "i4"),
    ("obs_dates", "u2"),
    ("gen_dates", "u2"),
    ("gen_seconds", "i4"),
    ("edge_s", "i4"),
    ("edge_w", "i4"),
    ("edge_n", "i4"),
    ("edge_e", "i4"),
    ("cx", "i4"),
    ("cy", "i4"),
    ("nx", "i4"),
    ("ny", "i4"),
    ("dx", "i4"),
    ("dy", "i4"),
    ("height", "u2"),
    ("compress","u2"),
    ("num_of_radars", "i4"),
    ("un_zip_bytes", "i4"),
    ("scale", "u2"),
    ("unused", "u2"),
    ("rgn_id", "8c"),
    ("units", "8c"),
    ("res", "60c"),
]

mocm_dtype = np.dtype(MOCM_HEADER)
