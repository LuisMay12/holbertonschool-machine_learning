#!/usr/bin/env python3
"""
Performs back propagation over a pooling layer.
"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer of a neural network.

    Args:
        dA: numpy.ndarray (m, h_new, w_new, c)
            Partial derivatives with respect to the
            output of the pooling layer.
        A_prev: numpy.ndarray (m, h_prev, w_prev, c)
            Output of the previous layer.
        kernel_shape: tuple (kh, kw)
            Size of the pooling kernel.
        stride: tuple (sh, sw)
            Strides for the pooling operation.
        mode: str, 'max' or 'avg'
            Pooling mode: maximum pooling or average pooling.

    Returns:
        numpy.ndarray: dA_prev, partial derivatives with respect to A_prev
    """
    m, h_new, w_new, c = dA.shape
    sh, sw = stride
    kh, kw = kernel_shape

    # Initialize derivatives array
    dA_prev = np.zeros(shape=A_prev.shape)

    for i in range(m):  # Examples (images)
        for h in range(h_new):  # heights
            for w in range(w_new):  # widths
                for f in range(c):  # channels
                    # Prepare slice indexes to account for stride
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw
                    # Update gradients for this channel
                    if mode == 'avg':
                        avg_dA = dA[i, h, w, f] / kh / kw
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            (np.ones((kh, kw)) * avg_dA)
                    elif mode == 'max':
                        region = A_prev[i, v_start:v_end, h_start:h_end, f]
                        mask = (region == np.max(region))
                        dA_prev[i, v_start:v_end, h_start:h_end, f] +=\
                            mask * dA[i, h, w, f]

    return dA_prev
