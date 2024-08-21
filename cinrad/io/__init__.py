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


def read_auto(filename: str) -> RadarBase:
    """Read radar data, auto detected type of file .
    Args:
        filename: file name of radar data.

    Radar types:

        1. StandardData, standard format based data.
        2. StandardPUP, standard PUP format data.
        3. MocMosaic, mosaic format data v3.0.
        4. SWAN, SWAN format data.
        5. CinradReader, cinrad format based data.
        6. PhasedArrayData, standard format phased array radar data(XAD-2023).
        TODO:PUP & CinradReader(fix later)
    """
    with prepare_file(filename) as file:
        flag = file.read(125)
    if flag[0:4] == b"RSTM":
        if flag[8:12] == b"\x01\x00\x00\x00":
            return StandardData(filename)
        elif flag[8:12] == b"\x02\x00\x00\x00":
            return StandardPUP(filename)
        elif flag[8:12] == b"\x10\x00\x00\x00":
            return StandardData(filename)
        else:
            raise Exception("Unknown standard radar type")
    elif flag[0:2] == b"\x01\x00":
        return PhasedArrayData(filename)
    elif flag[0:3] == b"MOC":
        return MocMosaic(filename)
    if flag[50:54] == b"SWAN":
        return SWAN(filename)
    sc_flag = flag[100:106]
    cc_flag = flag[116:122]
    if flag[14:16] == b"\x01\x00" or sc_flag == b"CINRAD" or cc_flag == b"CINRAD":
        return CinradReader(filename)
    raise Exception("Unknown radar type")
