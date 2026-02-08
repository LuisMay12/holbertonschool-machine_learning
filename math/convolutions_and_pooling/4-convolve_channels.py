#!/usr/bin/env python3
"""4-convolve_channels.py
Performs a convolution on images with channels.
"""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels.

    Args:
        images (np.ndarray): shape (m, h, w, c) containing multiple images.
        kernel (np.ndarray): shape (kh, kw, c) containing the kernel.
        padding (str or tuple): 'same', 'valid', or (ph, pw).
        stride (tuple): (sh, sw).

    Returns:
        np.ndarray: shape (m, oh, ow) containing the convolved images.
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride

    if kc != c:
        raise ValueError("kernel channels must match image channels")

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        error = "padding must be 'same', 'valid', or a tuple (ph, pw)"
        raise ValueError(error)

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((m, oh, ow))

    # Only two loops: over output spatial coordinates
    for i in range(oh):
        for j in range(ow):
            y = i * sh
            x = j * sw
            region = padded[:, y:y + kh, x:x + kw, :]   # (m, kh, kw, c)
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2, 3))

    return output
