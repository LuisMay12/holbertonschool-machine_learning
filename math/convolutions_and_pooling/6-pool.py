#!/usr/bin/env python3
"""6-pool.py
Performs pooling on images.
"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images.

    Args:
        images (np.ndarray): shape (m, h, w, c) containing multiple images.
        kernel_shape (tuple): (kh, kw) pooling window size.
        stride (tuple): (sh, sw) stride.
        mode (str): 'max' for max pooling, 'avg' for average pooling.

    Returns:
        np.ndarray: shape (m, oh, ow, c) containing pooled images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    oh = (h - kh) // sh + 1
    ow = (w - kw) // sw + 1

    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            y = i * sh
            x = j * sw
            region = images[:, y:y + kh, x:x + kw, :]  # (m, kh, kw, c)

            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))
            else:
                raise ValueError("mode must be 'max' or 'avg'")

    return output
