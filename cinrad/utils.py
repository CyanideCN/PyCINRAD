# -*- coding: utf-8 -*-
# Author: Puyuan Du

from typing import Union, Any

import numpy as np

from cinrad.constants import deg2rad, vil_const
from cinrad.projection import height
from cinrad._typing import Array_T, Number_T

try:
    from cinrad._utils import *
except ImportError:
    # When the C-extension doesn't exist, define the functions in Python.

    def r2z(r: np.ndarray) -> np.ndarray:
        return 10 ** (r / 10)

    def vert_integrated_liquid(
        ref: np.ndarray,
        distance: np.ndarray,
        elev: Array_T,
        beam_width: float = 0.99,
        threshold: Union[float, int] = 18.0,
        density: bool = False,
    ) -> np.ndarray:
        r"""
        Calculate vertically integrated liquid (VIL) in one full scan

        Parameters
        ----------
        ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
            reflectivity data
        distance: numpy.ndarray dim=2 (distance, azimuth)
            distance from radar site
        elev: numpy.ndarray or list dim=1
            elevation angles in degree
        threshold: float
            minimum reflectivity value to take into calculation

        Returns
        -------
        data: numpy.ndarray
            vertically integrated liquid data
        """
        if density:
            raise NotImplementedError("VIL density calculation is not implemented")
        v_beam_width = beam_width * deg2rad
        elev = np.array(elev) * deg2rad
        xshape, yshape = ref[0].shape
        distance *= 1000
        hi_arr = distance * np.sin(v_beam_width / 2)
        vil = _vil_iter(xshape, yshape, ref, distance, elev, hi_arr, threshold)
        return vil

    def _vil_iter(
        xshape: int,
        yshape: int,
        ref: np.ndarray,
        distance: np.ndarray,
        elev: Array_T,
        hi_arr: np.ndarray,
        threshold: Number_T,
    ) -> np.ndarray:
        # r = np.clip(ref, None, 55) #reduce the influence of hails
        r = ref
        z = r2z(r)
        VIL = np.zeros((xshape, yshape))
        for i in range(xshape):
            for j in range(yshape):
                vert_r = r[:, i, j]
                vert_z = z[:, i, j]
                dist = distance[i][j]
                position = np.where(vert_r > threshold)[0]
                if position.shape[0] == 0:
                    continue
                pos_s = position[0]
                pos_e = position[-1]
                m1 = 0
                hi = hi_arr[i][j]
                for l in range(pos_e):
                    ht = dist * (np.sin(elev[l + 1]) - np.sin(elev[l]))
                    factor = ((vert_z[l] + vert_z[l + 1]) / 2) ** (4 / 7)
                    m1 += vil_const * factor * ht
                mb = vil_const * vert_z[pos_s] ** (4 / 7) * hi
                mt = vil_const * vert_z[pos_e] ** (4 / 7) * hi
                VIL[i][j] = m1 + mb + mt
        return VIL

    def echo_top(
        ref: np.ndarray,
        distance: np.ndarray,
        elev: Array_T,
        radarheight: Number_T,
        threshold: Number_T = 18.0,
    ) -> np.ndarray:
        r"""
        Calculate height of echo tops (ET) in one full scan

        Parameters
        ----------
        ref: numpy.ndarray dim=3 (elevation angle, distance, azimuth)
            reflectivity data
        distance: numpy.ndarray dim=2 (distance, azimuth)
            distance from radar site
        elev: numpy.ndarray or list dim=1
            elevation angles in degree
        radarheight: int or float
            height of radar
        drange: float or int
            range of data to be calculated
        threshold: float
            minimum value of reflectivity to be taken into calculation

        Returns
        -------
        data: numpy.ndarray
            echo tops data
        """
        ref[np.isnan(ref)] = 0
        xshape, yshape = ref[0].shape
        et = np.zeros((xshape, yshape))
        h_ = list()
        for i in elev:
            h = height(distance, i, radarheight)
            h_.append(h)
        hght = np.concatenate(h_).reshape(ref.shape)
        for i in range(xshape):
            for j in range(yshape):
                vert_h = hght[:, i, j]
                vert_r = ref[:, i, j]
                if vert_r.max() < threshold:  # Vertical points don't satisfy threshold
                    et[i][j] = 0
                    continue
                elif vert_r[-1] >= threshold:  # Point in highest scan exceeds threshold
                    et[i][j] = vert_h[-1]
                    continue
                else:
                    position = np.where(vert_r >= threshold)[0]
                    if position[-1] == 0:
                        et[i][j] = vert_h[0]
                        continue
                    else:
                        pos = position[-1]
                        z1 = vert_r[pos]
                        z2 = vert_r[pos + 1]
                        h1 = vert_h[pos]
                        h2 = vert_h[pos + 1]
                        w1 = (z1 - threshold) / (z1 - z2)
                        w2 = 1 - w1
                        et[i][j] = w1 * h2 + w2 * h1
        return et


def potential_maximum_gust(et: np.ndarray, vil: np.ndarray) -> np.ndarray:
    r"""
    Estimate the potential maximum gust with a descending downdraft by Stewart's formula
    """
    return np.sqrt(20.628571 * vil - 2.3810964e-6 * et ** 2)


def potential_maximum_gust_from_reflectivity(
    ref: np.ndarray, distance: np.ndarray, elev: Array_T
) -> np.ndarray:
    et = echo_top(ref, distance, elev, 0)
    vil = vert_integrated_liquid(ref, distance, elev)
    return potential_maximum_gust(et, vil)


def lanczos_differentiator(winlen: int):
    # Copyright (c) 2011-2018, wradlib developers.
    m = (winlen - 1) / 2
    denom = m * (m + 1.0) * (2 * m + 1.0)
    k = np.arange(1, m + 1)
    f = 3 * k / denom
    return np.r_[f[::-1], [0], -f]


def kdp_from_phidp(
    phidp: np.ndarray, winlen: int = 7, dr: int = 1.0, method: bool = None
) -> np.ndarray:
    from scipy.stats import linregress
    from scipy.ndimage.filters import convolve1d

    # Copyright (c) 2011-2018, wradlib developers.
    """Retrieves :math:`K_{DP}` from :math:`Phi_{DP}`.
    In normal operation the method uses convolution to estimate :math:`K_{DP}`
    (the derivative of :math:`Phi_{DP}` with Low-noise Lanczos differentiators.
    The results are very similar to the fallback moving window linear
    regression (`method=slow`), but the former is *much* faster, depending on
    the percentage of NaN values in the beam, though.
    For further reading please see `Differentiation by integration using \
    orthogonal polynomials, a survey <https://arxiv.org/pdf/1102.5219>`_ \
    and `Low-noise Lanczos differentiators \
    <http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/\
    lanczos-low-noise-differentiators/>`_.
    The fast method provides fast :math:`K_{DP}` retrieval but will return NaNs
    in case at least one value in the moving window is NaN. The remaining gates
    are treated by using local linear regression where possible.
    Please note that the moving window size ``winlen`` is specified as the number of
    range gates. Thus, this argument might need adjustment in case the
    range resolution changes.
    In the original publication (:cite:`Vulpiani2012`), the value ``winlen=7``
    was chosen for a range resolution of 1km.
    Warning
    -------
    The function is designed for speed by allowing to process
    multiple dimensions in one step. For this purpose, the RANGE dimension
    needs to be the LAST dimension of the input array.
    Parameters
    ----------
    phidp : :class:`numpy:numpy.ndarray`
        multi-dimensional array, note that the range dimension must be the
        last dimension of the input array.
    winlen : int
        Width of the window (as number of range gates)
    dr : float
        gate length in km
    method : str
        If None uses fast convolution based differentiation, if 'slow' uses
        linear regression.
    Examples
    --------
    >>> import wradlib
    >>> import numpy as np
    >>> import matplotlib.pyplot as pl
    >>> pl.interactive(True)
    >>> kdp_true = np.sin(3 * np.arange(0, 10, 0.1))
    >>> phidp_true = np.cumsum(kdp_true)
    >>> phidp_raw = phidp_true + np.random.uniform(-1, 1, len(phidp_true))
    >>> gaps = np.concatenate([range(10, 20), range(30, 40), range(60, 80)])
    >>> phidp_raw[gaps] = np.nan
    >>> kdp_re = wradlib.dp.kdp_from_phidp(phidp_raw)
    >>> line1 = pl.plot(np.ma.masked_invalid(phidp_true), "b--", label="phidp_true")  # noqa
    >>> line2 = pl.plot(np.ma.masked_invalid(phidp_raw), "b-", label="phidp_raw")  # noqa
    >>> line3 = pl.plot(kdp_true, "g-", label="kdp_true")
    >>> line4 = pl.plot(np.ma.masked_invalid(kdp_re), "r-", label="kdp_reconstructed")  # noqa
    >>> lgnd = pl.legend(("phidp_true", "phidp_raw", "kdp_true", "kdp_reconstructed"))  # noqa
    >>> pl.show()
    """
    assert (
        winlen % 2
    ) == 1, "Window size N for function kdp_from_phidp must be an odd number."

    shape = phidp.shape
    phidp = phidp.reshape((-1, shape[-1]))

    # Make really sure winlen is an integer
    winlen = int(winlen)

    if method == "slow":
        kdp = np.zeros(phidp.shape) * np.nan
    else:
        window = lanczos_differentiator(winlen)
        kdp = convolve1d(phidp, window, axis=1)

    # find remaining NaN values with valid neighbours
    invalidkdp = np.isnan(kdp)
    if not np.any(invalidkdp.ravel()):
        # No NaN? Return KdP
        return kdp.reshape(shape) / 2.0 / dr

    # Otherwise continue
    x = np.arange(phidp.shape[-1])
    valids = ~np.isnan(phidp)
    kernel = np.ones(winlen, dtype="i4")
    # and do the slow moving window linear regression
    for beam in range(len(phidp)):
        # number of valid neighbours around one gate
        nvalid = np.convolve(valids[beam], kernel, "same") > winlen / 2
        # find those gates which have invalid Kdp AND enough valid neighbours
        nangates = np.where(invalidkdp[beam] & nvalid)[0]
        # now iterate over those
        for r in nangates:
            ix = np.arange(
                max(0, r - int(winlen / 2)), min(r + int(winlen / 2) + 1, shape[-1])
            )
            # check again (just to make sure...)
            if np.sum(valids[beam, ix]) < winlen / 2:
                # not enough valid values inside our window
                continue
            kdp[beam, r] = linregress(
                x[ix][valids[beam, ix]], phidp[beam, ix[valids[beam, ix]]]
            )[0]
        # take care of the start and end of the beam
        #   start
        ix = np.arange(0, winlen)
        if np.sum(valids[beam, ix]) >= 2:
            kdp[beam, 0 : int(winlen / 2)] = linregress(
                x[ix][valids[beam, ix]], phidp[beam, ix[valids[beam, ix]]]
            )[0]
        # end
        ix = np.arange(shape[-1] - winlen, shape[-1])
        if np.sum(valids[beam, ix]) >= 2:
            kdp[beam, -int(winlen / 2) :] = linregress(
                x[ix][valids[beam, ix]], phidp[beam, ix[valids[beam, ix]]]
            )[0]

    # accounting for forward/backward propagation AND gate length
    return kdp.reshape(shape) / 2.0 / dr
