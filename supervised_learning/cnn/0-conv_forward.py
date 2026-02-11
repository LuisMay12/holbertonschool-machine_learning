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
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Calculate output dimensions
    output_h = (h_prev + 2 * ph - kh) // sh + 1
    output_w = (w_prev + 2 * pw - kw) // sw + 1

    # Padding input as needed
    padded_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded input
            region = padded_A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(c_new):
                # Convolve each input (m) in the region, using kernel k
                convolved[:, i, j, k] = np.sum((region * W[:, :, :, k]),
                                               axis=(1, 2, 3))

    # Layer l activation output: A(l+1) = g(Z), with Z = A(l) * W + b
    return activation(convolved + b)
