#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer
    of a neural network.


    A_prev : numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        Activations from the previous layer.
    W : numpy.ndarray of shape (kh, kw, c_prev, c_new)
        Convolution kernels (filters).
    b : numpy.ndarray of shape (1, 1, 1, c_new)
        Biases for each output channel.
    activation : function
        Activation function applied to the convolution output.
    padding : str, "same" or "valid"
        Type of padding.
    stride : tuple (sh, sw)
        Stride for height and width.

    Returns
    -------
    numpy.ndarray:
    Activated output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev_w, c_new = W.shape
    sh, sw = stride

    if c_prev_w != c_prev:
        raise ValueError("W and A_prev channel dimensions do not match")

    if padding not in ("same", "valid"):
        raise ValueError('padding must be "same" or "valid"')

    # Compute padding
    if padding == "valid":
        ph = 0
        pw = 0
    else:
        # "same" padding: choose padding so output size is ceil(input/stride)
        h_out = int(np.ceil(h_prev / sh))
        w_out = int(np.ceil(w_prev / sw))

        ph = int(np.ceil(((h_out - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_out - 1) * sw + kw - w_prev) / 2))

    # Pad input
    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant",
        constant_values=0
    )

    # Output spatial dimensions
    h_out = ((h_prev + 2 * ph - kh) // sh) + 1
    w_out = ((w_prev + 2 * pw - kw) // sw) + 1

    Z = np.zeros((m, h_out, w_out, c_new))

    # Convolution
    for i in range(m):
        for y in range(h_out):
            y_start = y * sh
            y_end = y_start + kh
            for x in range(w_out):
                x_start = x * sw
                x_end = x_start + kw

                # (kh, kw, c_prev)
                window = A_pad[i, y_start:y_end, x_start:x_end, :]

                # Apply each filter
                for c in range(c_new):
                    sum = np.sum(window * W[:, :, :, c]) + b[0, 0, 0, c]
                    Z[i, y, x, c] = sum

    return activation(Z)
