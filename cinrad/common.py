# -*- coding: utf-8 -*-
# Author: Puyuan Du

from xarray import Dataset


def get_dtype(data: Dataset) -> str:
    all_data = list(data.keys())
    geo_var_name = ["longitude", "latitude", "height"]
    for i in geo_var_name:
        if i in all_data:
            all_data.remove(i)
    return all_data[0]
