cimport numpy as np
import numpy as np
ctypedef np.float64_t npfloat


def trapz(function, xs, dxs, output_shape):
    cdef int output_size = np.prod(output_shape)
    return np.asarray(_trapz(function, xs, dxs, output_size)).reshape(output_shape)


def _trapz(function, np.ndarray[npfloat, ndim=1] xs, np.ndarray[npfloat, ndim=1] dxs, int output_size):
    cdef np.ndarray[npfloat, ndim=1] integral = np.zeros(output_size, dtype=np.float64)
    cdef int i
    cdef float xi
    for i, xi in enumerate(xs):
        if i != 0:
            integral = integral + dxs[i] * (function(xs[i-1])+function(xi)) / 2
        else:
            continue
    return integral
