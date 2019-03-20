from typing import List, Union
from numpy  import ndarray

from cinrad.datastruct import Radial

GList = Union[list, ndarray]
RList = List[Radial]
boardcast_type = Union[int, float, ndarray]
number_type = Union[int, float]