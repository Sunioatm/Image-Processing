import numpy as np

def notch_filter(rows, cols, r, filter_type="low-pass"):
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), np.float32)

    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if filter_type == "low-pass":
                if distance <= r:
                    filter[x, y] = 1
            elif filter_type == "high-pass":
                if distance > r:
                    filter[x, y] = 1

    return filter