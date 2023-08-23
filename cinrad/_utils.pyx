# cython: language_level=3
cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sin

cdef double deg2rad, vil_const
cdef int rm = 8500

deg2rad = 3.141592653589793 / 180
vil_const = 3.44e-6

cdef np.ndarray height(np.ndarray distance, double elevation, double radarheight):
    # unit is meter
    return distance * sin(elevation * deg2rad) + distance ** 2 / (2 * rm) + radarheight

cdef double height_single(double distance, double elevation):
    return distance * sin(elevation * deg2rad) + distance ** 2 / (2 * rm)

@cython.boundscheck(False)
def vert_integrated_liquid(double[:, :, ::1] ref, double[:, ::1] distance, double[::1] elev,
                           double beam_width=0.99, double threshold=18., bint density=False):
    cdef double v_beam_width, m1, mb, mt, factor, ht, dist, r_tmp, h_higher, h_lower
    cdef long long[::1] position
    cdef Py_ssize_t xshape, yshape, zshape
    cdef int pos_s, pos_e, l, i, j, k
    cdef double[:] vert_r, vert_z
    v_beam_width = beam_width * deg2rad
    zshape, xshape, yshape = ref.shape[0], ref.shape[1], ref.shape[2]
    cdef double[:, ::1] VIL = np.zeros((xshape, yshape))
    for i in range(xshape):
        for j in range(yshape):
            vert_r = ref[:, i, j]
            vert_z = vert_r.copy()
            for k in range(zshape):
                vert_z[k] = 10 ** (vert_z[k] / 10)
            dist = distance[i][j] * 1000
            hi = dist * sin(v_beam_width / 2)
            pos_s = -1
            pos_e = -1
            for k in range(zshape):
                if vert_r[k] > threshold:
                    pos_e = k
                    if pos_s == -1:
                        pos_s = k
            if pos_s == -1:
                continue
            m1 = 0.
            for l in range(pos_e):
                ht = dist * (sin(elev[l + 1] * deg2rad) - sin(elev[l] * deg2rad))
                factor = ((vert_z[l] + vert_z[l + 1]) / 2) ** (4 / 7)
                m1 += vil_const * factor * ht
            if density == False:
                mb = vil_const * vert_z[pos_s] ** (4 / 7) * hi
                mt = vil_const * vert_z[pos_e] ** (4 / 7) * hi
                VIL[i][j] = m1 + mb + mt
            elif density == True:
                if pos_s == pos_e:
                    # If there's only one gate satisfying threshold, assigning VILD as zero
                    VIL[i][j] = 0
                else:
                    h_lower = height_single(dist / 1000, elev[pos_s])
                    h_higher = height_single(dist / 1000, elev[pos_e])
                    VIL[i][j] = m1 / (h_higher - h_lower)
    return np.asarray(VIL)

@cython.boundscheck(False)
def echo_top(double[:, :, ::1] ref, double[:, ::1] distance, double[::1] elev,
             double radarheight, double threshold=18.):
    cdef np.ndarray r, h, vert_h, vert_r, hght
    cdef int xshape, yshape, pos, i, j
    cdef list h_
    cdef double z1, z2, h1, h2, w1, w2, e
    cdef long long[::1] position
    xshape, yshape = ref.shape[1], ref.shape[2]
    cdef double[:, ::1] et = np.zeros((xshape, yshape))
    h_ = list()
    for e in elev:
        h = height(np.asarray(distance), e, radarheight)
        h_.append(h)
    hght = np.concatenate(h_).reshape(ref.shape[0], ref.shape[1], ref.shape[2])
    r = np.asarray(ref)
    r[np.isnan(ref)] = 0
    for i in range(xshape):
        for j in range(yshape):
            vert_h = hght[:, i, j]
            vert_r = r[:, i, j]
            if np.max(vert_r) < threshold: # Vertical points don't satisfy threshold
                et[i][j] = 0
                continue
            elif vert_r[-1] >= threshold: # Point in highest scan exceeds threshold
                et[i][j] = vert_h[-1]
                continue
            else:
                position = np.nonzero(vert_r >= threshold)[0]
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
    return np.asarray(et)