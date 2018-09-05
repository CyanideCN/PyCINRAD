import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float sin(float theta)

cdef _height(np.ndarray distance, float elevation, float radarheight):
    return distance * sin(elevation) + distance ** 2 / (2 * 8500) + radarheight / 1000

def echo_top(tuple data, list elev, float radarheight, float threshold):
    cdef int xshape, yshape, pos
    cdef np.ndarray h, h_mask, vertical, position
    cdef float h_pt, r_pt, h_pt_, z1, z2, h1, h2, w1, w2
    cdef list h_, vert_h, vert_r, vert_h_
    et = []
    r = np.ma.array(data[0], mask=(data[0] > threshold))
    xshape, yshape = r[0].shape
    h_ = list()
    for i in elev:
        h = _height(data[1], i, radarheight)
        h_.append(h)
    hght = np.concatenate(h_).reshape(r.shape)
    h_mask = hght * r.mask
    for i in xrange(xshape):
        for j in xrange(yshape):
            vert_h = list()
            vert_r = list()
            vert_h_ = list()
            for k in xrange(1, 10):
                #index from highest angle
                h_pt = h_mask[-1 * k][i][j]
                r_pt = data[0][-1 * k][i][j]
                h_pt_ = hght[-1 * k][i][j]
                vert_h.append(h_pt)
                vert_r.append(r_pt)
                vert_h_.append(h_pt_)
            vertical = np.array(vert_h)
            position = np.where(vertical > 0)[0]
            try:
                pos = position[0]
            except IndexError:#empty array
                et.append(0)
                continue
            if pos == 0:
                height = vertical[pos]
                et.append(height)
            else:
                try:
                    elev[pos - 1]
                except IndexError:
                    et.append(vertical[pos])
                    continue
                z1 = vert_r[pos]
                z2 = vert_r[pos - 1]
                h1 = vertical[pos]
                h2 = vert_h_[pos - 1]
                w1 = (z1 - threshold) / (z1 - z2)
                w2 = 1 - w1
                et.append(w1 * h2 + w2 * h1)
    return np.array(et).reshape(xshape, yshape)