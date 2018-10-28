# -*- coding: utf-8 -*-
# Author: Du puyuan

import matplotlib
from matplotlib import rcParams

matplotlib.use('Agg')

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['PingFang SC', 'Hiragino Sans GB', 'Heiti SC', 'Microsoft YaHei', '微软雅黑', 'SimHei',
                               'WenQuanYi Micro Hei']
from . import ppi
