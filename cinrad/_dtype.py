# -*- coding: utf-8 -*-
# Author: Puyuan Du

import numpy as np

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

# TODO: resolve incompatible size
# currently not used
_CC_HEADER = [('filetype', 'c', 16),
              ('country', 'c', 30),
              ('province', 'c', 20),
              ('station', 'c', 40),
              ('station_number', 'c', 10),
              ('radartype', 'c', 20),
              ('longitude', 'c', 16),
              ('latitude', 'c', 16),
              ('longitude_var', 'i4'),
              ('latitude_var', 'i4'),
              ('height', 'i4'),
              ('s_max_angle', 'i2'),
              ('s_opt_angle', 'i2'),
              ('ucsyear1', 'B'),
              ('ucsyear2', 'B'),
              ('ucsmonth', 'B'),
              ('ucsday', 'B'),
              ('ucsminute', 'B'),
              ('ucssecond', 'B'),
              ('uctimefrom', 'B'),
              ('uceyear1', 'B'),
              ('uceyear2', 'B'),
              ('ucemonth', 'B'),
              ('uceday', 'B'),
              ('ucehour', 'B'),
              ('uceminute', 'B'),
              ('ucesecond', 'B'),
              ('scanmode', 'B'),
              ('milisecond', 'u4'),
              ('rhi_a', 'u2'),
              ('rhi_l', 'i2'),
              ('rhi_h', 'i2'),
              ('echo_type', 'u2'),
              ('prod_code', 'u2'),
              ('calibration', 'c'),
              ('remain', 'c', 3),
              ('remain2', 'c', 660),
              ('antennag', 'i4'),
              ('power', 'i4'),
              ('wavelength', 'i4'),
              ('beam_h', 'u2'),
              ('beam_l', 'u2'),
              ('polarization', 'u2'),
              ('log_a', 'u2'),
              ('line_a', 'u2'),
              ('agcp', 'u2'),
              ('freq_mode', 'u2'),
              ('freq_repeat', 'u2'),
              ('ppp_pulse', 'u2'),
              ('fft_point', 'u2'),
              ('process_type', 'u2'),
              ('clutter_t', 'B'),
              ('sidelobe', 'c'),
              ('velocity_t', 'B'),
              ('filder_p', 'B'),
              ('noise_t', 'B'),
              ('sqi_t', 'B'),
              ('intensity_c', 'B'),
              ('intensity_r', 'B'),
              ('cal_noise', 'B'),
              ('cal_power', 'B'),
              ('cal_pulse_width', 'B'),
              ('cal_work_freq', 'B'),
              ('cal_log', 'B'),
              ('remain3', 'c', 92),
              ('data_offset', 'i4')]

_CC_VPPI_SCAN_PARAMETER = [('maxv', 'u2'),
                           ('maxl', 'u2'),
                           ('bindwidth', 'u2'),
                           ('binnumber', 'u2'),
                           ('recordnumber', 'u2'),
                           ('v_rotate', 'u2'),
                           ('prf1', 'u2'),
                           ('prf2', 'u2'),
                           ('spulsew', 'u2'),
                           ('elevation_angle', 'u2'),
                           ('sweepstatus', 'B'),
                           ('ambiguousp', 'B')]

_CC_DATA = [('Z', 'i2', 500),
            ('V', 'i2', 500),
            ('W', 'i2', 500)]

_CD_RADARSITE = [('country', 'c', 30),
                 ('province', 'c', 20),
                 ('station', 'c', 40),
                 ('stationid', 'c', 10),
                 ('radartype', 'c', 20),
                 ('longitude', 'c', 16),
                 ('latitude', 'c', 16),
                 ('longitude_var', 'i4'),
                 ('latitude_var', 'i4'),
                 ('height', 'i4'),
                 ('max_angle', 'i2'),
                 ('op_angle', 'i2'),
                 ('mang_freq', 'i2')]

_CD_PERFORMANCE_PARAM = [('antenna_g', 'i4'),
                         ('beam_h', 'u2'),
                         ('beam_l', 'u2'),
                         ('polarization', 'B'),
                         ('sidelobe', 'c'),
                         ('power', 'i4'),
                         ('wavelength', 'i4'),
                         ('log_a', 'u2'),
                         ('line_a', 'u2'),
                         ('agcp', 'u2'),
                         ('clutter_t', 'B'),
                         ('velocity_p', 'B'),
                         ('filder_p', 'B'),
                         ('noise_t', 'B'),
                         ('sqit', 'B'),
                         ('intensity_c', 'B'),
                         ('intensity_r', 'B')]

_CD_LAYER_PARAM = [('ambiguous_p', 'B'),
                   ('a_rotate', 'u2'),
                   ('prf_1', 'u2'),
                   ('prf_2', 'u2'),
                   ('s_pulse_w', 'u2'),
                   ('max_v', 'u2'),
                   ('max_l', 'u2'),
                   ('bin_width', 'u2'),
                   ('bin_number', 'u2'),
                   ('record_number', 'u2'),
                   ('elevation', 'i2')]

_CD_OBSERVATION_PARAM = [('s_type', 'B'),
                         ('s_year', 'u2'),
                         ('s_month', 'B'),
                         ('s_day', 'B'),
                         ('s_hour', 'B'),
                         ('s_minute', 'B'),
                         ('time_p', 'B'),
                         ('s_milisecond', 'u4'),
                         ('calibration', 'B'),
                         ('intensity_i', 'B'),
                         ('velocity_p', 'B'),
                         ('layer_param', np.dtype(_CD_LAYER_PARAM)),
                         ('rhi_a', 'u2'),
                         ('rhi_l', 'i2'),
                         ('rhi_h', 'i2'),
                         ('e_year', 'u2'),
                         ('e_month', 'B'),
                         ('e_day', 'B'),
                         ('e_hour', 'B'),
                         ('e_minute', 'B'),
                         ('e_second', 'B'),
                         ('e_tenth', 'B')]

_CD_DATA = [('m_dbz', 'B'),
            ('m_vel', 'c'),
            ('m_undbz', 'B'),
            ('m_sw', 'B')]

_CD_DATARECORD = [('start_az', 'u2'),
                  ('start_el', 'u2'),
                  ('end_az', 'u2'),
                  ('end_el', 'u2'),
                  ('data', np.dtype(_CD_DATA))]

_CD_HEADER = [('radar_site', np.dtype(_CD_RADARSITE)),
              ('radar_performance', np.dtype(_CD_PERFORMANCE_PARAM)),
              ('radar_observation', np.dtype(_CD_OBSERVATION_PARAM)),
              ('res', 'c', 163)]

SAB_dtype = np.dtype(_S_HEADER + _S_INFO + _SAB_DATA)
CAB_dtype = np.dtype(_S_HEADER + _S_INFO + _CAB_DATA)
CC_param = np.dtype(_CC_VPPI_SCAN_PARAMETER)
CC_data = np.dtype(_CC_DATA)
