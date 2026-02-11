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
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for y in range(h_new):
            y_start = y * sh
            y_end = y_start + kh
            for x in range(w_new):
                x_start = x * sw
                x_end = x_start + kw

                if mode == 'avg':
                    # Distribute gradient evenly across the window
                    da = dA[i, y, x, :]  # (c,)
                    da = da.reshape((1, 1, 1, c))
                    avg = da / (kh * kw)
                    dA_prev[i, y_start:y_end, x_start:x_end, :] += avg

                elif mode == 'max':
                    # Pass gradient to the max location(s) in the window
                    # (kh, kw, c)
                    window = A_prev[i, y_start:y_end, x_start:x_end, :]
                    # (1, 1, c)
                    max_vals = np.max(window, axis=(0, 1), keepdims=True)
                    mask = (window == max_vals)  # (kh, kw, c) boolean

                    da = dA[i, y, x, :].reshape((1, 1, c))  # (1, 1, c)
                    dA_prev[i, y_start:y_end, x_start:x_end, :] += mask * da

    return dA_prev
