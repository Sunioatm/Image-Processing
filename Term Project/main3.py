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

    # Try different parameters for Canny edge detection
    for threshold1 in range(50, 150, 10):
        for threshold2 in range(150, 250, 10):
            edges = cv2.Canny(blurred, threshold1, threshold2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            logo_contour = max(contours, key=cv2.contourArea)

            # Save the segmentation result
            output_file = f"{output_path}_t1_{threshold1}_t2_{threshold2}.jpg"
            cv2.imwrite(output_file, logo_contour)
            

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
