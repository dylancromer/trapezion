cimport numpy as np
import numpy as np
ctypedef np.float64_t DTYPE


def trapz(function, xs, dxs, output_shape):
    cdef int output_size = np.prod(output_shape)
    return np.asarray(_trapz(function, xs, dxs, output_size)).reshape(output_shape)


def _trapz(function, DTYPE[:] xs, DTYPE[:] dxs, int output_size):
    cdef DTYPE[::1] integral = np.zeros(output_size, dtype=np.float64)
    cdef int i
    cdef DTYPE[:] f_i_minus_1
    cdef DTYPE[:] f_i
    cdef DTYPE dxi
    for i in range(len(xs)):
        if i != 0:
            f_i = function(xs[i])
            f_i_minus_1 = function(xs[i-1])
            dxi = dxs[i]
            for j in range(output_size):
                integral[j] = integral[j] + dxi * (f_i_minus_1[j]+f_i[j]) / 2
        else:
            continue
    return integral
