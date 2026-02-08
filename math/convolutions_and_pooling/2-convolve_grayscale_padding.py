#!/usr/bin/env python3
"""2-convolve_grayscale_padding.py
Performs a convolution on grayscale images with custom padding.
"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding.

    Args:
        images (np.ndarray): shape (m, h, w) containing m grayscale images.
        kernel (np.ndarray): shape (kh, kw) containing the kernel.
        padding (tuple): (ph, pw) padding sizes for height and width.

    Returns:
        np.ndarray: shape (m, oh, ow) containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    padded = np.pad(
        images,
        pad_width=((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    output = np.zeros((m, oh, ow))

    # Only two loops: over output spatial coordinates
    for i in range(oh):
        for j in range(ow):
            window = padded[:, i:i + kh, j:j + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
