import numpy as np
from scipy.ndimage.measurements import center_of_mass


def argmax2d(data: np.ndarray) -> (int, int):
    return np.unravel_index(np.argmax(data), data.shape)


def subpixel_argmax2d(heatmap: np.ndarray,
                      window_size=10):
    w, h = argmax2d(heatmap)

    window_size = min(window_size, h, w, np.min(np.array(heatmap.shape) - [w, h] - 1))

    w1, w2 = w - window_size, w + window_size + 1
    h1, h2 = h - window_size, h + window_size + 1

    crop = heatmap[w1:w2, h1:h2]
    y, x = center_of_mass(crop)
    return w - window_size + y, h - window_size + x
