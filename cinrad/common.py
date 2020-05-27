# -*- coding: utf-8 -*-
# Author: Puyuan Du

from xarray import Dataset


def get_dtype(data: Dataset) -> str:
    all_data = list(data.keys())
    geo_var_name = ["longitude", "latitude", "height", "x_cor", "y_cor", "RF"]
    for i in geo_var_name:
        if i in all_data:
            all_data.remove(i)
    return all_data[0]


def is_radial(data: Dataset) -> bool:
    coords = set(data.coords.keys())
    return coords == set(("distance", "azimuth"))
