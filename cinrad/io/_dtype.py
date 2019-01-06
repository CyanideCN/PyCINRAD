# -*- coding: utf-8 -*-
# Author: Puyuan Du

import numpy as np

def gen_CC():
    from ._radar_struct.CC import scan_param_dtype, header_dtype, data_dtype
    return scan_param, header_dtype, data_dtype

CC_param, CC_header, CC_data = gen_CC()

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
