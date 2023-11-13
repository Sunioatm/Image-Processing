import cv2
import numpy as np

def segment_oranges_rgb(image, lower_rgb, upper_rgb):
    # Create a mask for orange color in RGB space
    mask = cv2.inRange(image, lower_rgb, upper_rgb)

    # Bitwise-AND mask and original image to extract orange regions
    orange_parts = cv2.bitwise_and(image, image, mask=mask)

    # Create an inverse mask for non-orange parts
    inverse_mask = cv2.bitwise_not(mask)

    # Create a full-size image with blue color
    blue_color = np.full(image.shape, [255, 0, 0], dtype=np.uint8)  # BGR format for blue

    # Apply inverse mask to the blue image
    non_orange_parts = cv2.bitwise_and(blue_color, blue_color, mask=inverse_mask)

    # Combine the orange and blue parts
    segmented_image = cv2.add(orange_parts, non_orange_parts)

    return segmented_image

# Load the image
image = cv2.imread('oranges.jpg')  # Replace with your image path

# Define the RGB range for orange
# You might need to adjust these values based on the specific shade of orange
lower_orange_rgb = np.array([0, 40, 100])  # Example lower bound
upper_orange_rgb = np.array([50, 150, 255])  # Example upper bound

# Segment the oranges
segmented_image = segment_oranges_rgb(image, lower_orange_rgb, upper_orange_rgb)

# Display the result
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
