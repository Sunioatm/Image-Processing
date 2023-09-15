import numpy as np

def power_law(image, c, gamma):
    enhanced_image = np.power(image / 255.0, gamma)
    enhanced_image = np.uint8(255 * c * enhanced_image)
    return enhanced_image
