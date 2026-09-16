from pathlib import Path

import numpy as np
from xarray import Dataset, DataArray

from cinrad.correct.dealias import dealias, dealias_unwrap_2d
from align_cases import vel_nanmask

GOLDEN = Path(__file__).parent / "golden" / "cython_ref.npz"


def _golden(name):
    with np.load(GOLDEN) as data:
        return data[name]


def test_dealias_dataset_nan_mask():
    vdata, nyq = vel_nanmask()
    ds = Dataset(
        {"VEL": DataArray(vdata, dims=["azimuth", "distance"])},
        attrs={"nyquist_vel": nyq},
    )
    out = np.asarray(dealias(ds)["VEL"].values)
    assert np.array_equal(np.isnan(out), np.isnan(vdata))
    finite = ~np.isnan(vdata)
    assert np.array_equal(out[finite], dealias_unwrap_2d(vdata, nyq)[finite])
    assert np.array_equal(dealias_unwrap_2d(vdata, nyq), _golden("dealias_nanmask"))
