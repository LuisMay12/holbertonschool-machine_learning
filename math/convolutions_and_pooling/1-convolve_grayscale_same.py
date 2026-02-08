#!/usr/bin/env python3
"""1-convolve_grayscale_same.py
Performs a same convolution on grayscale images.
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images.

    Args:
        images (np.ndarray): shape (m, h, w) containing m grayscale images.
        kernel (np.ndarray): shape (kh, kw) containing the kernel.

    Returns:
        np.ndarray: shape (m, h, w) with convolved images (same padding).
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # For "same" convolution we want output height/width equal to input.
    # The needed total padding can be computed and split:
    # top/bottom, left/right.
    ph = kh - 1
    pw = kw - 1

    # when padding is odd (even-sized kernels), put the "extra" pad on TOP/LEFT
    ph_top = (ph + 1) // 2
    ph_bottom = ph // 2
    pw_left = (pw + 1) // 2
    pw_right = pw // 2

    padded = np.pad(
        images,
        pad_width=((0, 0), (ph_top, ph_bottom), (pw_left, pw_right)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros((m, h, w))

    # Only two loops: over the output spatial positions (i, j)
    for i in range(h):
        for j in range(w):
            window = padded[:, i:i + kh, j:j + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
