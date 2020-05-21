# -*- coding: utf-8 -*-
# Author: Puyuan Du
from typing import List, Union
from numpy import ndarray
from xarray import Dataset

Array_T = Union[list, ndarray]
Volume_T = List[Dataset]
Boardcast_T = Union[int, float, ndarray]
Number_T = Union[int, float]
