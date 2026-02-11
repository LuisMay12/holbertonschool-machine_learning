#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer.
"""

import numpy as np


def conv_backkward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray (m, h_new, w_new, c_new)
            Derivatives with respect to the unactivated output Z.
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev)
            Output of the previous layer (input to this conv layer).
        W: numpy.ndarray (kh, kw, c_prev, c_new)
            Kernels (filters).
        b: numpy.ndarray (1, 1, 1, c_new)
            Biases.
        padding: "same" or "valid"
        stride: (sh, sw)

    Returns:
        dA_prev: partial derivatives with respect to A_prev
        dW: partial derivatives with respect to W
        db: partial derivatives with respect to b
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev_w != c_prev:
        raise ValueError("W and A_prev channel dimensions do not match")
    if padding not in ("same", "valid"):
        raise ValueError('padding must be "same" or "valid"')

    # Padding amounts (match your conv_forward: floor division + symmetric pad)
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph = 0
        pw = 0

    # Pad A_prev and initialize gradients
    A_pad = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                   mode="constant")
    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)

    # Bias gradient: sum over batch and spatial dims
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    h_new, w_new = dZ.shape[1], dZ.shape[2]

    # Backprop through convolution
    for i in range(m):
        for y in range(h_new):
            y_start = y * sh
            y_end = y_start + kh
            for x in range(w_new):
                x_start = x * sw
                x_end = x_start + kw

                # (kh, kw, c_prev)
                a_slice = A_pad[i, y_start:y_end, x_start:x_end, :]

                for c in range(c_new):
                    dz = dZ[i, y, x, c]

                    # dW accumulates input slice scaled by dz
                    dW[:, :, :, c] += a_slice * dz

                    # dA accumulates filter
                    # scaled by dz
                    scaled = W[:, :, :, c] * dz
                    dA_pad[i, y_start:y_end, x_start:x_end, :] += scaled

    # Unpad dA to match A_prev shape
    if ph == 0 and pw == 0:
        dA_prev = dA_pad
    else:
        dA_prev = dA_pad[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
