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


def oks(y_true: np.ndarray,
        y_pred: np.ndarray,
        input_size: int,
        k=None):
    y_true = np.array(y_true, dtype='float64')
    y_pred = np.array(y_pred, dtype='float64')
    scale = np.sqrt(np.sum(np.power(input_size, 2))) / 100
    scales = np.array([scale for _ in range(y_true.shape[0])])

    if k is None:
        k = np.ones(y_true.shape[0])
    else:
        k = np.array(k)

    distances = np.sum((y_true - y_pred) ** 2, axis=1)
    return np.exp(-distances / (2 * (scales * k) ** 2))
