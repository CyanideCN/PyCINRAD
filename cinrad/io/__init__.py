"""Classes for decoding level 2 and level 3 radar data.
"""

from cinrad.io.base import RadarBase, prepare_file
from cinrad.io.level2 import *
from cinrad.io.level3 import *
from typing import Union

RadarBase_T = Union[
    RadarBase, StandardData, CinradReader, StandardPUP, MocMosaic, SWAN, PhasedArrayData, PUP
]


def read_level2(filename: str) -> RadarBase:
    """Read CINRAD level 2 data."""
    with prepare_file(filename) as file:
        magic = file.read(4)
    if magic == b"RSTM":
        return StandardData(filename)
    else:
        return CinradReader(filename)


def read_auto(
    filename: str,
    **kwargs
) -> RadarBase_T:
    """Read radar data, auto detected type of file .

    Args:
        filename: file name of radar data.
        **kwargs: keyword arguments.
        eg: `SWAN(, product="CR")``CinradReader(, radar_type="CC")`

    Radar types:

        1. StandardData, standard format based data.

        2. StandardPUP, standard PUP format data.

        3. MocMosaic, mosaic format data v3.0.

        4. SWAN, SWAN format data.

        5. CinradReader, cinrad format based data.

        6. PhasedArrayData, standard format phased array radar data(XAD-2023).

        7: PUP, NEXRAD Level 3 (NIDS) product files.

    """
    with prepare_file(filename) as file:
        flag = file.read(125)
    if flag[:4] == b"RSTM":
        if flag[8:12] == b"\x01\x00\x00\x00":
            return StandardData(filename)
        elif flag[8:12] == b"\x02\x00\x00\x00":
            return StandardPUP(filename)
        elif flag[8:12] == b"\x10\x00\x00\x00":
            return PhasedArrayData(filename)
        else:
            raise Exception("Unknown standard radar type")
    elif flag[:3] == b"MOC":
        return MocMosaic(filename)
    elif flag[50:54] == b"SWAN":
        return SWAN(filename, **kwargs)
    elif flag[:4] == b"NXUS" or flag[:4] == b"SDUS" or flag[:4] == b"NOUS":
        return PUP(filename)
    sc_flag = flag[100:106]
    cc_flag = flag[116:122]
    if flag[14:16] == b"\x01\x00" or sc_flag == b"CINRAD" or cc_flag == b"CINRAD":
        return CinradReader(filename, **kwargs)
    raise Exception("Unknown radar type")
