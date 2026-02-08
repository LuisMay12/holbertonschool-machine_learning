#!/usr/bin/env python3
"""0-convolve_grayscale_valid.py
Performs a valid convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    Args:
        images (np.ndarray): shape (m, h, w) containing m grayscale images.
        kernel (np.ndarray): shape (kh, kw) containing the kernel.

    Returns:
        np.ndarray: shape (m, h - kh + 1, w - kw + 1) with convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    oh = h - kh + 1
    ow = w - kw + 1

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            window = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
