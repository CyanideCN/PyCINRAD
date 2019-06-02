from typing import List, Union
from numpy  import ndarray

from cinrad.datastruct import Radial

Array_T = Union[list, ndarray]
Volume_T = List[Radial]
Boardcast_T = Union[int, float, ndarray]
Number_T = Union[int, float]
