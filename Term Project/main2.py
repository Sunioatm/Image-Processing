import cv2
import numpy as np

images = ['handbag1.jpg', 'handbag2.jpeg', 'handbag3.jpg', 'handbag4.jpg', 'handbag7.jpg', 'handbag8.jpeg']

def segment_logo_with_canny_and_contours(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume the largest contour in the image is the logo
    # This assumption needs validation for each specific case
    logo_contour = max(contours, key=cv2.contourArea)

    # Create an empty mask
    mask = np.zeros_like(gray)

    # Draw the filled contour on the mask
    cv2.drawContours(mask, [logo_contour], -1, 255, -1)

    # Create a solid color background
    background = np.full_like(image, (0, 0, 255))

    # Bitwise operations to isolate the logo and apply the background
    out = np.where(mask[:, :, None].astype(bool), image, background)

    # Save the output image
    cv2.imwrite(output_path, out)

    return out

# Define paths
for image in images:
    input_image_path = image
    output_image_path = f"segmented_{image}"

    # Segment the logo
    segmented_logo = segment_logo_with_canny_and_contours(input_image_path, output_image_path)

    # Check if the segmentation was successful
    if segmented_logo is not None:
        print("Segmentation completed. The output image is saved as:", output_image_path)
    else:
        print("Segmentation was not successful.")
