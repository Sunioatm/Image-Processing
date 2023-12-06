import cv2
import numpy as np

def find_logo_by_otsu(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Placeholder for logo detection logic
    # This needs to be refined based on the characteristics of the logo and handbag
    logo_contour = max(contours, key=cv2.contourArea)

    return logo_contour

def segment_logo(input_image_path, output_image_path):
    # Load the image
    image = cv2.imread(input_image_path)
    if image is None:
        print("Image not found")
        return

    # Find logo contour using Otsu's method
    logo_contour = find_logo_by_otsu(image)

    # Create a mask for the logo
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [logo_contour], -1, (255), thickness=cv2.FILLED)

    # Segment the logo
    segmented_logo = cv2.bitwise_and(image, image, mask=mask)

    # Create a solid color background and blend with the segmented logo
    background = np.full(image.shape, (0, 0, 255), np.uint8)  # Solid red background
    masked_background = cv2.bitwise_and(background, background, mask=~mask)
    result = cv2.add(segmented_logo, masked_background)

    # Save the output image
    cv2.imwrite(output_image_path, result)

    print("Logo segmentation completed.")

# Example usage
segment_logo('handbag1.jpg', 'segmented_logo.jpg')
