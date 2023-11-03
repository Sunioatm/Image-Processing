import numpy as np

def gaussian_lpf(rows, cols, D0):
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols, 2), np.float32)
    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            filter[x, y] = np.exp(-(distance ** 2) / (2 * (D0 ** 2)))
    return filter

def gaussian_hpf(rows, cols, D0):
    return 1 - gaussian_lpf(rows, cols, D0)