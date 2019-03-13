# -*- coding: utf-8 -*-
# Author: Puyuan Du

import numpy as np

__all__ = ['CC_param', 'CC_header', 'CC_data', 'SDD_header', 'SDD_site', 'SDD_task',
           'SDD_cut', 'SDD_rad_header', 'SDD_mom_header', 'SAB_dtype', 'CAB_dtype',
           'swan_header_dtype']

from cinrad.io._radar_struct.CC import scan_param_dtype as CC_param, header_dtype as CC_header, data_dtype as CC_data
from cinrad.io._radar_struct.standard_data import (generic_header_dtype as SDD_header, site_config_dtype as SDD_site,
                                                   task_config_dtype as SDD_task, cut_config_dtype as SDD_cut,
                                                   radial_header_dtype as SDD_rad_header, moment_header_dtype as SDD_mom_header)

_S_HEADER = [('spare', 'u2', 7),
             ('a', 'u2'),
             ('res', 'u2', 6)]

_S_INFO = [('time', 'u4'),
           ('day', 'u2'),
           ('unambiguous_distance', 'u2'),
           ('azimuth', 'u2'),
           ('radial_num', 'u2'),
           ('radial_state', 'u2'),
           ('elevation', 'u2'),
           ('el_num', 'u2'),
           ('first_gate_r', 'u2'),
           ('first_gate_v', 'u2'),
           ('gate_length_r', 'u2'),
           ('gate_length_v', 'u2'),
           ('gate_num_r', 'u2'),
           ('gate_num_v', 'u2'),
           ('sector_num', 'u2'),
           ('system_coff', 'u4'),
           ('r_pointer', 'u2'),
           ('v_pointer', 'u2'),
           ('w_pointer', 'u2'),
           ('v_reso', 'u2'),
           ('vcp_mode', 'u2'),
           ('res2', 'u2', 4),
           ('r_pointer_2', 'u2'),
           ('v_pointer_2', 'u2'),
           ('w_pointer_2', 'u2'),
           ('nyquist_vel', 'u2'),
           ('res3', 'u2', 19)]

_SAB_DATA = [('r', 'u1', 460),
             ('v', 'u1', 920),
             ('w', 'u1', 920),
             ('res4', 'u2', 2)]

_CAB_DATA = [('r', 'u1', 800),
             ('v', 'u1', 1600),
             ('w', 'u1', 1600),
             ('res4', 'u2', 2)]

SAB_dtype = np.dtype(_S_HEADER + _S_INFO + _SAB_DATA)
CAB_dtype = np.dtype(_S_HEADER + _S_INFO + _CAB_DATA)

swan_header = [('data_type', '12c'),
               ('data_name', '38c'),
               ('name', '8c'),
               ('version', '8c'),
               ('year', 'u2'),
               ('month', 'u2'),
               ('day', 'u2'),
               ('hour', 'u2'),
               ('minute', 'u2'),
               ('interval', 'u2'),
               ('x_grid_num', 'u2'),
               ('y_grid_num', 'u2'),
               ('z_grid_num', 'u2'),
               ('radar_count', 'i4'),
               ('start_lon', 'f4'),
               ('start_lat', 'f4'),
               ('center_lon', 'f4'),
               ('center_lat', 'f4'),
               ('x_reso', 'f4'),
               ('y_reso', 'f4'),
               ('height', '40f4'),
               ('station_names', '20c16'),
               ('station_lons', '20f4'),
               ('station_lats', '20f4'),
               ('station_alts', '20f4'),
               ('mosaic_flags', '20B'),
               ('m_data_type', 'i2'),
               ('dimension', 'i2'),
               ('res', '168c')]

swan_header_dtype = np.dtype(swan_header)