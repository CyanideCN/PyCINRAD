# Fixture factories for Cython alignment tests. Numpy only, no cinrad imports.
import numpy as np


def _c64(*a, **k):
    return np.ascontiguousarray(np.asarray(*a, **k), dtype=np.float64)


def _polar_dist(nrange, naz, r0=0.25, r1=230.0):
    dist = np.empty((nrange, naz), dtype=np.float64)
    dist[:, :] = np.linspace(r0, r1, nrange)[:, None]
    return dist


def volume_small():
    ref = np.arange(0, 27, 1, dtype=np.float64).reshape(3, 3, 3)
    dist = np.broadcast_to(np.arange(0, 3, dtype=np.float64), (3, 3)).copy()
    elev = np.arange(0, 3, dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_mid(dist_offset=0.0):
    ref = np.arange(0, 64, 1, dtype=np.float64).reshape(4, 4, 4)
    dist = np.broadcast_to(np.arange(0, 4, dtype=np.float64), (4, 4)).copy() + dist_offset
    elev = np.arange(0, 4, dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_cinrad():
    rng = np.random.default_rng(42)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5], dtype=np.float64)
    dist = _polar_dist(80, 360)
    ref = rng.normal(25, 15, size=(len(elev), 80, 360)).astype(np.float64)
    ref[ref < 0] = 0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_below():
    ref = np.full((5, 10, 12), 5.0, dtype=np.float64)
    dist = np.broadcast_to(np.arange(1, 11, dtype=np.float64)[:, None], (10, 12)).copy()
    elev = np.linspace(0.5, 6, 5, dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_above():
    ref, dist, elev = volume_below()
    return np.full_like(ref, 40.0), dist, elev


def volume_single():
    ref = np.full((1, 8, 8), 25.0, dtype=np.float64)
    dist = np.broadcast_to(np.arange(1, 9, dtype=np.float64)[:, None], (8, 8)).copy()
    elev = np.array([0.5], dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_nan():
    ref, dist, elev = volume_mid(1.0)
    ref = ref.copy()
    ref[1, 2, 2] = np.nan
    ref[0, 0, :] = np.nan
    return _c64(ref), _c64(dist), _c64(elev)


def volume_mix():
    ref = np.zeros((6, 16, 20), dtype=np.float64)
    ref[2:5, 4:12, 3:15] = 30
    ref[3, 8, 8] = 50
    ref[0, :, :] = 10
    dist = np.broadcast_to(np.linspace(1, 100, 16)[:, None], (16, 20)).copy()
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0, 9.9], dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_zeros():
    ref = np.zeros((3, 2, 2), dtype=np.float64)
    dist = np.ones((2, 2), dtype=np.float64)
    elev = np.array([0.5, 1.5, 2.4], dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_sa230():
    rng = np.random.default_rng(20240916)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5], dtype=np.float64)
    dist = _polar_dist(230, 360)
    ref = rng.normal(22, 18, size=(len(elev), 230, 360))
    ref = np.clip(ref, 0, 75).astype(np.float64)
    storm = (np.arange(230)[:, None] - 80) ** 2 / 400 + (np.arange(360) - 120) ** 2 / 900
    ref[:, storm < 1] += 25
    return _c64(ref), _c64(dist), _c64(elev)


def volume_cc512():
    rng = np.random.default_rng(7)
    elev = np.array(
        [0.5, 1.5, 2.4, 3.3, 4.3, 5.2, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5],
        dtype=np.float64,
    )
    dist = _polar_dist(120, 512, 0.3, 150)
    ref = rng.normal(20, 12, size=(len(elev), 120, 512))
    ref = np.clip(ref, -5, 65).astype(np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_irregular():
    rng = np.random.default_rng(99)
    elev = np.array([0.5, 0.8, 2.0, 5.5, 7.0, 15.2, 19.5], dtype=np.float64)
    dist = _polar_dist(60, 90, 1.0, 100)
    ref = rng.uniform(0, 45, size=(len(elev), 60, 90)).astype(np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_thresh():
    # exactly / just below / just above VIL (>) and ET (>=) thresholds of 18
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0], dtype=np.float64)
    dist = _polar_dist(12, 16, 2.0, 80)
    ref = np.full((5, 12, 16), 10.0, dtype=np.float64)
    ref[:, :, 0] = 18.0
    ref[:, :, 1] = 17.999999999
    ref[:, :, 2] = 18.000000001
    ref[0, :, 3] = 18.0
    ref[2, :, 4] = 18.0
    ref[-1, :, 5] = 18.0
    ref[-1, :, 6] = 19.0
    ref[1:4, 5:8, 7] = 18.0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_hail():
    rng = np.random.default_rng(13)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9, 14.6, 19.5], dtype=np.float64)
    dist = _polar_dist(40, 72, 1.0, 80)
    ref = rng.normal(30, 8, size=(len(elev), 40, 72))
    ref[2:6, 10:18, 20:35] = rng.uniform(55, 75, size=(4, 8, 15))
    return _c64(ref.astype(np.float64)), _c64(dist), _c64(elev)


def volume_vild_single():
    # only one gate above 18 in each column that has echo
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0, 9.9], dtype=np.float64)
    dist = _polar_dist(20, 24, 5.0, 100)
    ref = np.full((6, 20, 24), 5.0, dtype=np.float64)
    for j in range(24):
        k = j % 6
        ref[k, :, j] = 25.0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_et_top():
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0], dtype=np.float64)
    dist = _polar_dist(16, 20, 3.0, 90)
    ref = np.full((5, 16, 20), 10.0, dtype=np.float64)
    ref[-1] = 25.0
    ref[-1, 2, 3] = 17.0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_et_pos0():
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0], dtype=np.float64)
    dist = _polar_dist(16, 20, 3.0, 90)
    ref = np.full((5, 16, 20), 10.0, dtype=np.float64)
    ref[0] = 25.0
    ref[1:] = 5.0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_et_interp():
    # last exceeding gate at several k, so interpolation uses pos and pos+1
    elev = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 6.0, 9.9], dtype=np.float64)
    dist = _polar_dist(18, 28, 4.0, 110)
    ref = np.full((7, 18, 28), 5.0, dtype=np.float64)
    for j in range(28):
        k = 1 + (j % 5)  # 1..5, never last tilt
        ref[: k + 1, :, j] = 20.0 + 0.3 * np.arange(k + 1)[:, None]
        ref[k + 1, :, j] = 10.0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_nan_mix():
    rng = np.random.default_rng(123)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0, 9.9, 14.6], dtype=np.float64)
    dist = _polar_dist(50, 80, 1.0, 150)
    ref = rng.normal(24, 14, size=(len(elev), 50, 80)).astype(np.float64)
    mask = rng.random(ref.shape) < 0.08
    ref[mask] = np.nan
    return _c64(ref), _c64(dist), _c64(elev)


def volume_gap():
    # first and last exceed with holes in the middle (VIL still sums l in range(pos_e))
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0, 9.9], dtype=np.float64)
    dist = _polar_dist(24, 30, 2.0, 120)
    ref = np.full((6, 24, 30), 5.0, dtype=np.float64)
    ref[0] = 22.0
    ref[2] = 8.0
    ref[4] = 30.0
    ref[5] = 12.0
    return _c64(ref), _c64(dist), _c64(elev)


def volume_neg():
    rng = np.random.default_rng(5)
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0], dtype=np.float64)
    dist = _polar_dist(24, 32, 2.0, 80)
    ref = rng.normal(5, 20, size=(len(elev), 24, 32)).astype(np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_exact18():
    elev = np.array([0.5, 1.5, 2.4, 3.4, 6.0], dtype=np.float64)
    dist = _polar_dist(16, 20, 3.0, 90)
    ref = np.full((5, 16, 20), 18.0, dtype=np.float64)
    return _c64(ref), _c64(dist), _c64(elev)


def volume_sparse():
    elev = np.array([0.5, 1.5, 2.4, 6.0, 9.9, 14.6], dtype=np.float64)
    dist = _polar_dist(30, 40, 1.0, 120)
    ref = np.full((6, 30, 40), 0.0, dtype=np.float64)
    ref[0, 5, 7] = 40.0
    ref[5, 5, 7] = 22.0
    ref[:, 10, 10] = 25.0
    ref[3, 20, :] = 30.0
    return _c64(ref), _c64(dist), _c64(elev)


VOLUME_CASES = {
    "small": volume_small,
    "mid": volume_mid,
    "mid_d1": lambda: volume_mid(1.0),
    "cinrad": volume_cinrad,
    "below": volume_below,
    "above": volume_above,
    "single": volume_single,
    "nan": volume_nan,
    "mix": volume_mix,
    "zeros": volume_zeros,
    "sa230": volume_sa230,
    "cc512": volume_cc512,
    "irregular": volume_irregular,
    "thresh": volume_thresh,
    "hail": volume_hail,
    "vild_single": volume_vild_single,
    "et_top": volume_et_top,
    "et_pos0": volume_et_pos0,
    "et_interp": volume_et_interp,
    "nan_mix": volume_nan_mix,
    "gap": volume_gap,
    "neg": volume_neg,
    "exact18": volume_exact18,
    "sparse": volume_sparse,
}

# density=True hits ZeroDivisionError in original Cython when distance==0
VILD_SKIP = {"mid"}


def vel_simple():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36, endpoint=False),
        np.linspace(1, 100, 40),
        indexing="ij",
    )
    true_v = 10 * np.sin(az) * np.exp(-((rng - 50) / 40) ** 2)
    wrapped = np.angle(np.exp(1j * true_v * np.pi / nyq)) * nyq / np.pi
    return _c64(wrapped), nyq


def vel_folded():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 36, endpoint=False),
        np.linspace(1, 100, 40),
        indexing="ij",
    )
    true_v = 40 * np.sin(az) * (rng / 100)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return _c64(wrapped), nyq


def vel_nanmask():
    wrapped, nyq = vel_folded()
    wrapped = wrapped.copy()
    wrapped[::5, ::7] = np.nan
    wrapped[10:12, :] = np.nan
    return _c64(wrapped), nyq


def vel_cinrad():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 360, endpoint=False),
        np.linspace(0.25, 230, 230),
        indexing="ij",
    )
    true_v = 35 * np.sin(2 * az) * np.tanh(rng / 40)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return _c64(wrapped), nyq


def vel_tiny():
    return _c64([[1.0, 2.0], [3.0, -20.0]]), 27.0


def vel_sa460():
    nyq = 27.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 360, endpoint=False),
        np.linspace(0.25, 230, 460),
        indexing="ij",
    )
    true_v = 48 * np.sin(az) * np.exp(-((rng - 80) / 70) ** 2) + 12 * np.cos(3 * az)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return _c64(wrapped), nyq


def vel_islands():
    nyq = 27.0
    v = np.zeros((80, 100), dtype=np.float64)
    v[5:25, 10:40] = ((35 * np.linspace(-1, 1, 20)[:, None] + nyq) % (2 * nyq)) - nyq
    v[50:70, 55:90] = ((-40 * np.linspace(0, 1, 20)[:, None] + nyq) % (2 * nyq)) - nyq
    v[30:35, :] = np.nan
    return _c64(v), nyq


def vel_nyq11():
    nyq = 11.0
    az, rng = np.meshgrid(
        np.linspace(0, 2 * np.pi, 180, endpoint=False),
        np.linspace(1, 150, 120),
        indexing="ij",
    )
    true_v = 22 * np.sin(az) * np.tanh(rng / 30)
    wrapped = ((true_v + nyq) % (2 * nyq)) - nyq
    return _c64(wrapped), nyq


def vel_stripe_nan():
    wrapped, nyq = vel_folded()
    wrapped = wrapped.copy()
    wrapped[::3, :] = np.nan
    wrapped[:, ::11] = np.nan
    return _c64(wrapped), nyq


def vel_nyqedge():
    # stay inside (-nyq, nyq); exactly ±nyq is ±π and unwrap branch is ambiguous
    nyq = 27.0
    v = np.zeros((48, 64), dtype=np.float64)
    v[:, :] = 0.999 * nyq
    v[::2] = -0.999 * nyq
    v[10:16, 20:30] = 0.0
    v[0, 0] = np.nan
    return _c64(v), nyq


def vel_const():
    return _c64(np.full((36, 80), 5.0)), 27.0


def vel_checker():
    nyq = 16.0
    v = np.zeros((64, 96), dtype=np.float64)
    v[::2, ::2] = 12.0
    v[1::2, 1::2] = -12.0
    v[::4] = np.nan
    return _c64(v), nyq


VEL_CASES = {
    "simple": vel_simple,
    "folded": vel_folded,
    "nanmask": vel_nanmask,
    "cinrad": vel_cinrad,
    "tiny": vel_tiny,
    "sa460": vel_sa460,
    "islands": vel_islands,
    "nyq11": vel_nyq11,
    "stripe_nan": vel_stripe_nan,
    "nyqedge": vel_nyqedge,
    "const": vel_const,
    "checker": vel_checker,
}

# extra (threshold, beam, radarheight) variants: name -> (volume_key, kwargs)
VOLUME_VARIANTS = {
    "thresh40": ("thresh", {"threshold": 40.0}),
    "thresh10": ("irregular", {"threshold": 10.0}),
    "bw1": ("mix", {"beam_width": 1.0}),
    "h973": ("et_interp", {"radarheight": 973.0}),
}
