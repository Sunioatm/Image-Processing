import numpy as np

def map_to_gray_levels(image, levels):
    step = 256 // (levels)
    new_image = (image // step) * step
    return new_image.astype(np.uint8)