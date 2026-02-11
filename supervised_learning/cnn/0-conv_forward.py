#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a
    neural network.

    Args:
        A_prev: numpy.ndarray (m, h_prev, w_prev, c_prev)
        W: numpy.ndarray (kh, kw, c_prev, c_new)
        b: numpy.ndarray (1, 1, 1, c_new)
        activation: activation function to apply to the convolution output
        padding: "same" or "valid"
        stride: (sh, sw)

    Returns:
        numpy.ndarray: activated output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev_w != c_prev:
        raise ValueError("W and A_prev channel dimensions do not match")
    if padding not in ("same", "valid"):
        raise ValueError('padding must be "same" or "valid"')

    # Compute padding amounts (total ph, pw)
    if padding == "valid":
        ph = 0
        pw = 0
    else:
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    # If padding is odd, put the "extra" pad on TOP and LEFT
    ph_top = (ph + 1) // 2
    ph_bottom = ph // 2
    pw_left = (pw + 1) // 2
    pw_right = pw // 2

    # Pad the input
    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_left, pw_right), (0, 0)),
        mode="constant",
        constant_values=0
    )

    # Output dimensions
    h_out = ((h_prev + ph_top + ph_bottom - kh) // sh) + 1
    w_out = ((w_prev + pw_left + pw_right - kw) // sw) + 1

    Z = np.zeros((m, h_out, w_out, c_new))

    # Convolution operation
    for i in range(m):
        for y in range(h_out):
            y_start = y * sh
            y_end = y_start + kh
            for x in range(w_out):
                x_start = x * sw
                x_end = x_start + kw

                # (kh, kw, c_prev)
                window = A_pad[i, y_start:y_end, x_start:x_end, :]

                for c in range(c_new):
                    sum = np.sum(window * W[:, :, :, c]) + b[0, 0, 0, c]
                    Z[i, y, x, c] = sum

    return activation(Z)
