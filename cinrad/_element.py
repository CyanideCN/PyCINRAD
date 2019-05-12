# -*- coding: utf-8 -*-
# Author: Puyuan Du

import numpy as np

from cinrad.constants import *

__all__ = ['norm_plot', 'norm_cbar', 'cmap_plot', 'cmap_cbar', 'sec_plot', 'prodname', 'unit',
           'cbar_text']

norm_plot = {'REF':norm1, 'VEL':norm2, 'CR':norm1, 'ET':norm5, 'VIL':norm1, 'RF':norm3,
             'ZDR':norm6, 'PHI':norm7, 'RHO':norm8, 'TREF':norm1, 'KDP':norm9, 'VILD':norm10} # Normalize object used to plot
norm_cbar = {'REF':norm1, 'VEL':norm4, 'CR':norm1, 'ET':norm4, 'VIL':norm4,
             'ZDR':norm4, 'PHI':norm4, 'RHO':norm4, 'TREF':norm1, 'KDP':norm4,
             'VILD':norm4} # Normalize object used for colorbar
cmap_plot = {'REF':r_cmap, 'VEL':v_cmap, 'CR':r_cmap, 'ET':et_cmap, 'VIL':vil_cmap, 'RF':rf_cmap,
             'ZDR':zdr_cmap, 'PHI':kdp_cmap, 'RHO':cc_cmap, 'TREF':r_cmap, 'KDP':kdp_cmap,
             'VILD':vil_cmap}
cmap_cbar = {'REF':r_cmap, 'VEL':v_cbar, 'CR':r_cmap, 'ET':et_cbar, 'VIL':vil_cbar,
             'ZDR':zdr_cbar, 'PHI':kdp_cbar, 'RHO':cc_cbar, 'TREF':r_cmap, 'KDP':kdp_cbar,
             'VILD':vil_cbar}
sec_plot = {'REF':r_cmap_smooth, 'VEL':v_cmap_smooth, 'ZDR':zdr_cmap_smooth, 'PHI':kdp_cmap_smooth, 'RHO':cc_cmap_smooth,
            'KDP':kdp_cmap_smooth}
prodname = {'REF':'Base Reflectivity', 'VEL':'Base Velocity', 'CR':'Composite Ref.',
            'ET':'Echo Tops', 'VIL':'V Integrated Liquid', 'ZDR':'Differential Ref.',
            'PHI':'Differential Phase', 'RHO':'Correlation Coe.', 'TREF':'Total Reflectivity',
            'KDP':'Spec. Diff. Phase', 'VILD':'VIL Density'}
unit = {'REF':'dBz', 'VEL':'m/s', 'CR':'dBz', 'ET':'km', 'VIL':'kg/m**2', 'ZDR':'dB', 'PHI':'deg',
        'RHO':'', 'TREF':'dBz', 'KDP':'deg/km', 'VILD':'g/m**3'}
cbar_text = {'REF':None, 'VEL':['RF', '', '27', '20', '15', '10', '5', '1', '0',
                                '-1', '-5', '-10', '-15', '-20', '-27', '-35'],
             'CR':None, 'ET':['', '21', '20', '18', '17', '15', '14', '12',
                              '11', '9', '8', '6', '5', '3', '2', '0'],
             'VIL':['', '70', '65', '60', '55', '50', '45', '40', '35', '30',
                    '25', '20', '15', '10', '5', '0'],
             'ZDR':['', '5', '4', '3.5', '3', '2.5', '2', '1.5', '1', '0.8', '0.5',
                    '0.2', '0', '-1', '-2', '-3', '-4'],
             'PHI':np.linspace(360, 260, 17).astype(str),
             'RHO':['', '0.99', '0.98', '0.97', '0.96', '0.95', '0.94', '0.92', '0.9',
                    '0.85', '0.8', '0.7', '0.6', '0.5', '0.3', '0.1', '0'],
             'TREF':None, 'KDP':['', '20', '7', '3.1', '2.4', '1.7', '1.1', '0.75', '0.5',
                                 '0.33', '0.22', '0.15', '0.1', '-0.1', '-0.2', '-0.4', '-0.8'],
             'VILD':['', '6', '5', '4', '3.5', '3', '2.5', '2.1', '1.8', '1.5', '1.2',
                     '0.9', '0.7', '0.5', '0.3', '0.1']}