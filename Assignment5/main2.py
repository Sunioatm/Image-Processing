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

        return hue, saturation, intensity
    

def segment_oranges(image, hue_range):

    # Convert RGB to HSI
    hue, saturation, intensity = RGB_TO_HSI(image)

    # Create a mask based on hue, saturation, and intensity
    hue_mask = (hue >= hue_range[0]) & (hue <= hue_range[1])

    # Combine the masks
    mask = hue_mask 

    # Change non-orange parts to blue
    image[mask == False] = [255, 0, 0]  # BGR format for blue

    return image

# Example ranges (you will need to adjust these based on your image and analysis)
lower_orange_hue = 0.1  # Example value in radians
upper_orange_hue = 1  # Example value in radians

# Segment the oranges

image = cv2.imread("oranges.jpg")
segmented_image = segment_oranges(image, (lower_orange_hue, upper_orange_hue))

cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_folder = "./color_slicing"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
cv2.imwrite(os.path.join(output_folder,"segmented_oranges.jpg"), segmented_image)
