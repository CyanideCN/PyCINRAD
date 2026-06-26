"""
Numba-accelerated sweep-based CAPPI kernel.
When numba is available, this module provides ~20-50× speedup over the
pure-Python implementation in calc.CAPPI.
"""
import math

import numpy as np

__all__ = ["cappi_sweep_interp", "HAS_NUMBA"]

try:
    from numba import njit, prange
    from numba.typed import List as NumbaList
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


if HAS_NUMBA:
    # ----------------------------------------------------------------
    # JIT helper: linear interpolation with NaN handling
    # ----------------------------------------------------------------
    @njit
    def _nb_interp_linear(coord, coord_0, coord_1, dat_0, dat_1, fillvalue):
        if coord_1 == coord_0:
            return fillvalue
        n0 = math.isnan(dat_0)
        n1 = math.isnan(dat_1)
        if n0 and n1:
            return fillvalue
        if n0:
            return dat_1
        if n1:
            return dat_0
        return ((coord_1 - coord) * dat_0 + (coord - coord_0) * dat_1) / (
            coord_1 - coord_0
        )

    # ----------------------------------------------------------------
    # JIT helper: bilinear interpolation within a single sweep
    # ----------------------------------------------------------------
    @njit
    def _nb_interp_sweep_sample(
        azimuth, ranges, data, target_az_deg, target_r_m, fillvalue
    ):
        """
        Bilinear (azimuth × range) interpolation within one sweep.

        Parameters
        ----------
        azimuth : 1D float64, sorted (radians)
        ranges : 1D float64, sorted (km)
        data : 2D float64 (naz, nrange)
        target_az_deg : float, azimuth (degrees)
        target_r_m : float, slant range (meters)
        fillvalue : float
        """
        naz = len(azimuth)
        nr = len(ranges)
        if naz < 2 or nr < 2:
            return fillvalue

        r_km = target_r_m / 1000.0
        if r_km > ranges[-1] or r_km < ranges[0]:
            return fillvalue

        # --- azimuth binary search ---
        az_rad = target_az_deg * math.pi / 180.0
        iaz = 0
        for j in range(naz):
            if azimuth[j] > az_rad:
                iaz = j
                break
        else:
            iaz = naz

        if iaz >= naz:
            # wrap around
            iaz = 0
            az_rad_val = az_rad - 2.0 * math.pi
        else:
            az_rad_val = az_rad

        if iaz == 0:
            az_last = azimuth[naz - 1] - 2.0 * math.pi
        else:
            az_last = azimuth[iaz - 1]

        # --- range binary search ---
        ir = 0
        for j in range(nr):
            if ranges[j] > r_km:
                ir = j
                break
        else:
            ir = nr

        if ir <= 0:
            ir = 1
        elif ir >= nr:
            ir = nr - 1

        # --- bilinear interpolation ---
        er0 = _nb_interp_linear(
            az_rad_val, az_last, azimuth[iaz],
            data[iaz - 1, ir - 1], data[iaz, ir - 1], fillvalue,
        )
        er1 = _nb_interp_linear(
            az_rad_val, az_last, azimuth[iaz],
            data[iaz - 1, ir], data[iaz, ir], fillvalue,
        )
        return _nb_interp_linear(
            r_km, ranges[ir - 1], ranges[ir], er0, er1, fillvalue
        )

    # ----------------------------------------------------------------
    # Main JIT kernel: Cartesian → Antenna + sweep-based interpolation
    # for ALL grid points in parallel
    # ----------------------------------------------------------------
    @njit(parallel=True)
    def _cappi_kernel(
        x, y, level_height, h_radar, reff,
        # sweep metadata
        sweep_elevs,  # 1D float64, ne elements
        # sweep data (list of arrays)
        sweep_az_list,  # list of 1D float64
        sweep_r_list,  # list of 1D float64
        sweep_data_list,  # list of 2D float64
        fillvalue, beam_half,
    ):
        """
        Numba-parallel sweep-based CAPPI computation.

        Parameters
        ----------
        All array arguments are float64.
        x, y : 1D array, Cartesian coords (meters)
        level_height : float, target height (meters MSL)
        h_radar : float, radar altitude (meters MSL)
        reff : float, effective earth radius (meters)
        sweep_elevs : 1D array, elevation angles per sweep (degrees)
        sweep_az_list : list of 1D arrays, azimuths per sweep (radians)
        sweep_r_list : list of 1D arrays, ranges per sweep (km)
        sweep_data_list : list of 2D arrays, data per sweep (naz×nrange)
        fillvalue : float
        beam_half : float, half beam width (degrees)

        Returns
        -------
        1D float64 array, same length as x/y
        """
        n = len(x)
        ne = len(sweep_elevs)
        result = np.full(n, fillvalue, dtype=np.float64)

        for i in prange(n):
            xi, yi = x[i], y[i]

            # ---- Cartesian to Antenna (spherical geometry) ----
            s = math.sqrt(xi * xi + yi * yi)
            phi = s / reff
            r_radar = reff + h_radar
            r_gate = reff + level_height

            r_sq = (r_radar * r_radar + r_gate * r_gate
                    - 2.0 * r_radar * r_gate * math.cos(phi))
            if r_sq < 0.0:
                r_sq = 0.0
            rng = math.sqrt(r_sq)

            # elevation
            if rng < 1e-10:
                el = 0.0
            else:
                cos_arg = ((r_radar * r_radar + rng * rng - r_gate * r_gate)
                           / (2.0 * r_radar * rng))
                if cos_arg > 1.0:
                    cos_arg = 1.0
                elif cos_arg < -1.0:
                    cos_arg = -1.0
                el = math.degrees(math.acos(cos_arg) - math.pi / 2.0)

            # azimuth (0–360 deg, north = 0, clockwise)
            az = math.degrees(math.pi / 2.0 - math.atan2(yi, xi))
            if az < 0.0:
                az += 360.0

            # ---- sweep-based interpolation ----
            if ne == 1:
                if abs(el - sweep_elevs[0]) > beam_half:
                    continue
                result[i] = _nb_interp_sweep_sample(
                    sweep_az_list[0], sweep_r_list[0], sweep_data_list[0],
                    az, rng, fillvalue,
                )
                continue

            # blind zone
            e0 = sweep_elevs[0]
            e1 = sweep_elevs[ne - 1]
            if el < e0 - beam_half or el > e1 + beam_half:
                continue

            if el <= e0 + beam_half:
                result[i] = _nb_interp_sweep_sample(
                    sweep_az_list[0], sweep_r_list[0], sweep_data_list[0],
                    az, rng, fillvalue,
                )
            elif el >= e1 - beam_half:
                result[i] = _nb_interp_sweep_sample(
                    sweep_az_list[ne - 1], sweep_r_list[ne - 1],
                    sweep_data_list[ne - 1], az, rng, fillvalue,
                )
            else:
                # find bracketing sweeps
                ie = ne - 1
                for j in range(ne):
                    if sweep_elevs[j] > el:
                        ie = j
                        break
                lower = ie - 1
                upper = ie

                val_low = _nb_interp_sweep_sample(
                    sweep_az_list[lower], sweep_r_list[lower],
                    sweep_data_list[lower], az, rng, fillvalue,
                )
                val_up = _nb_interp_sweep_sample(
                    sweep_az_list[upper], sweep_r_list[upper],
                    sweep_data_list[upper], az, rng, fillvalue,
                )

                if math.isnan(val_low) and math.isnan(val_up):
                    result[i] = fillvalue
                elif math.isnan(val_low):
                    result[i] = val_up
                elif math.isnan(val_up):
                    result[i] = val_low
                else:
                    w = (el - sweep_elevs[lower]) / (
                        sweep_elevs[upper] - sweep_elevs[lower]
                    )
                    result[i] = val_low * (1.0 - w) + val_up * w

        return result

    # ----------------------------------------------------------------
    # Public API: prepare data and call kernel
    # ----------------------------------------------------------------
    def cappi_sweep_interp(
        x, y, level_height, h_radar, reff,
        sweep_azimuths, sweep_ranges, sweep_data, sweep_elevs,
        fillvalue=np.nan, beam_half=0.5,
    ):
        """
        Numba-accelerated CAPPI computation.

        Parameters
        ----------
        x, y : 1D float64 array, Cartesian coords (m)
        level_height : float, target height (m MSL)
        h_radar : float, radar altitude (m MSL)
        reff : float, effective earth radius (m)
        sweep_azimuths : list of 1D float64, azimuths per sweep (rad, sorted)
        sweep_ranges : list of 1D float64, ranges per sweep (km, sorted)
        sweep_data : list of 2D float64, data per sweep (naz, nrange)
        sweep_elevs : 1D float64, elevation per sweep (deg)
        fillvalue : float
        beam_half : float, half beam width (deg)

        Returns
        -------
        1D float64 array, interpolated CAPPI values
        """
        # Build numba typed lists
        az_list = NumbaList()
        r_list = NumbaList()
        data_list = NumbaList()

        for az in sweep_azimuths:
            az_list.append(np.ascontiguousarray(az, dtype=np.float64))
        for r in sweep_ranges:
            r_list.append(np.ascontiguousarray(r, dtype=np.float64))
        for d in sweep_data:
            data_list.append(np.ascontiguousarray(d, dtype=np.float64))

        elevs = np.ascontiguousarray(sweep_elevs, dtype=np.float64)
        xc = np.ascontiguousarray(x, dtype=np.float64)
        yc = np.ascontiguousarray(y, dtype=np.float64)

        return _cappi_kernel(
            xc, yc, level_height, h_radar, reff,
            elevs, az_list, r_list, data_list,
            fillvalue, beam_half,
        )
