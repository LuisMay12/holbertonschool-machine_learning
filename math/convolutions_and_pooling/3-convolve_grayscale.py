#!/usr/bin/env python3
"""3-convolve_grayscale.py
Performs a convolution on grayscale images with padding and stride.
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images.

    Args:
        images (np.ndarray): shape (m, h, w) containing grayscale images.
        kernel (np.ndarray): shape (kh, kw) containing the kernel.
        padding (str or tuple): 'same', 'valid', or (ph, pw).
        stride (tuple): (sh, sw).

    Returns:
        np.ndarray: containing the convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # Determine padding
    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        # target output size:
        # round up
        oh = int(np.ceil(h / sh))
        ow = int(np.ceil(w / sw))

        # Solve for padding from:
        # oh = floor((h + 2ph - kh)/sh) + 1
        # => 2ph = (oh - 1)*sh + kh - h
        ph_total = max((oh - 1) * sh + kh - h, 0)
        pw_total = max((ow - 1) * sw + kw - w, 0)

        # put extra on TOP/LEFT when odd
        ph = (ph_total + 1) // 2
        pw = (pw_total + 1) // 2
        ph_bottom = ph_total // 2
        pw_right = pw_total // 2

        padded = np.pad(
            images,
            pad_width=((0, 0), (ph, ph_bottom), (pw, pw_right)),
            mode='constant',
            constant_values=0
        )
    else:
        error = "padding must be 'same','valid', or a tuple (ph, pw)"
        raise ValueError(error)

    # If not 'same' (where we already padded), pad symmetrically here
    if padding != 'same':
        padded = np.pad(
            images,
            pad_width=((0, 0), (ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0
        )

    h_p, w_p = padded.shape[1], padded.shape[2]

    oh = (h_p - kh) // sh + 1
    ow = (w_p - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    # Only two loops over output spatial coordinates
    for i in range(oh):
        for j in range(ow):
            y = i * sh
            x = j * sw
            window = padded[:, y:y + kh, x:x + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
