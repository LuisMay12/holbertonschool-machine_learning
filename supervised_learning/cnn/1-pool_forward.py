#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer.
"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network.

    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev)
            Output of the previous layer.
        kernel_shape: tuple (kh, kw)
            Size of the pooling kernel.
        stride: tuple (sh, sw)
            Strides for the pooling operation.
        mode: str, 'max' or 'avg'
            Pooling mode: maximum pooling or average pooling.

    Returns:
        numpy.ndarray: output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions (no padding in this task)
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    output = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            region = A_prev[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output
