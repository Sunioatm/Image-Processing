import numpy as np
import cv2
import os

def RGB_TO_HSI(img):
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Split image into RGB components
        b, g, r = cv2.split(img)

        # Calculate Intensity
        intensity = np.divide(b + g + r, 3)

        # Calculate Saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        saturation = 1 - 3 * np.divide(min_rgb, r + g + b + 1e-6)  # Add a small epsilon to avoid division by zero

        # Calculate Hue
        numerator = ((r - g) + (r - b)) / 2.0
        denominator = np.sqrt((r - g)**2 + (r - b) * (g - b))
        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-6, denominator)
        hue = np.arccos(numerator / denominator)
        
        # Adjust hue range to be between 0 and 2*pi
        hue[b > g] = 2 * np.pi - hue[b > g]

        # Merge all three channels into one HSI image
        hsi_img = cv2.merge((hue, saturation, intensity))

        return hsi_img

img = cv2.imread("fruit.jpg")
print(img[2])
img_hsi = RGB_TO_HSI(img)
print(img_hsi[1])