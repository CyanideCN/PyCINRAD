#cython: language_level=3
cimport numpy as np
import numpy as np

cdef double deg2rad, vil_const
cdef int rm

deg2rad = 3.141592653589793 / 180
rm = 8500
vil_const = 3.44e-6

cdef height(np.ndarray distance, double elevation, double radarheight):
    return distance * np.sin(elevation * deg2rad) + distance ** 2 / (2 * rm) + radarheight / 1000

cdef r2z(np.ndarray r):
    return 10 ** (r / 10)

def vert_integrated_liquid(np.ndarray ref, np.ndarray distance, list elev, beam_width=0.99, threshold=18.):
    cdef double v_beam_width, m1, mb, mt, factor, ht, dist
    cdef np.ndarray hi_arr, r, z, VIL, position, elev_
    cdef int xshape, yshape, pos_s, pos_e
    v_beam_width = beam_width * deg2rad
    elev_ = np.array(elev) * deg2rad
    xshape, yshape = ref[0].shape
    distance *= 1000
    hi_arr = distance * np.sin(v_beam_width / 2)
    #r = np.clip(ref, None, 55) #reduce the influence of hails
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
            m1 = 0.
            hi = hi_arr[i][j]
            for l in position[:-1].astype(int):
                ht = dist * (np.sin(elev_[l + 1]) - np.sin(elev_[l]))
                factor = ((vert_z[l] + vert_z[l + 1]) / 2) ** (4 / 7)
                m1 += vil_const * factor * ht
            mb = vil_const * vert_z[pos_s] ** (4 / 7) * hi
            mt = vil_const * vert_z[pos_e] ** (4 / 7) * hi
            VIL[i][j] = m1 + mb + mt
    return VIL

def echo_top(np.ndarray ref, np.ndarray distance, list elev, double radarheight, threshold=18.):
    cdef np.ndarray r, et, h, vert_h, vert_r
    cdef int xshape, yshape, pos
    cdef list h_
    cdef double z1, z2, h1, h2, w1, w2
    r = np.ma.array(ref, mask=(ref > threshold))
    xshape, yshape = r[0].shape
    et = np.zeros((xshape, yshape))
    h_ = list()
    for i in elev:
        h = height(distance, i, radarheight)
        h_.append(h)
    hght = np.concatenate(h_).reshape(r.shape[0], r.shape[1], r.shape[2])
    for i in range(xshape):
        for j in range(yshape):
            vert_h = hght[:, i, j]
            vert_r = ref[:, i, j]
            if vert_r.max() < threshold: # Vertical points don't satisfy threshold
                et[i][j] = 0
                continue
            elif vert_r[-1] >= threshold: # Point in highest scan exceeds threshold
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