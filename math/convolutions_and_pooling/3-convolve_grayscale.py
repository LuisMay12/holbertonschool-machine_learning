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

    if padding == 'valid':
        pt = pb = pl = pr = 0

    elif padding == 'same':
        # target output size:
        # round up
        oh = int(np.ceil(h / sh))
        ow = int(np.ceil(w / sw))

        # Total padding needed to achieve those output sizes:
        # oh = floor((h + ph_total - kh)/sh) + 1  where ph_total = pt + pb
        ph_total = max((oh - 1) * sh + kh - h, 0)
        pw_total = max((ow - 1) * sw + kw - w, 0)

        # Holberton alignment: extra padding goes to TOP/LEFT when odd
        pt = (ph_total + 1) // 2
        pb = ph_total // 2
        pl = (pw_total + 1) // 2
        pr = pw_total // 2

    else:
        # custom (ph, pw) padding per side, symmetric
        ph, pw = padding
        pt = pb = ph
        pl = pr = pw

    # --- Pad the images ---
    padded = np.pad(
        images,
        pad_width=((0, 0), (pt, pb), (pl, pr)),
        mode='constant',
        constant_values=0
    )

    # --- Output dimensions ---
    h_p, w_p = padded.shape[1], padded.shape[2]
    oh = (h_p - kh) // sh + 1
    ow = (w_p - kw) // sw + 1

    output = np.zeros((m, oh, ow))

    # Only two loops: i (rows), j (cols)
    for i in range(oh):
        for j in range(ow):
            y = i * sh
            x = j * sw
            window = padded[:, y:y + kh, x:x + kw]  # (m, kh, kw)
            output[:, i, j] = np.sum(window * kernel, axis=(1, 2))

    return output
