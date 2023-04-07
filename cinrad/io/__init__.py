"""Classes for decoding level 2 and level 3 radar data.
"""

from cinrad.io.base import RadarBase, prepare_file
from cinrad.io.level2 import *
from cinrad.io.level3 import *


def read_level2(filename: str) -> RadarBase:
    """Read CINRAD level 2 data."""
    with prepare_file(filename) as file:
        magic = file.read(4)
    if magic == b"RSTM":
        return StandardData(filename)
    else:
        return CinradReader(filename)
