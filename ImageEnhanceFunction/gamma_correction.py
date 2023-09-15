# power-law without constant
import numpy as np

def gamma_correction(image, gamma):
    enhanced_image = np.power(image / 255.0, gamma)
    enhanced_image = np.uint8(255 * enhanced_image)
    return enhanced_image