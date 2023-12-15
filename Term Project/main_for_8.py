import cv2
import numpy as np
from matplotlib import pyplot as plt

def is_logo_candidate(contour, img):
    min_area = 0.01 * img.shape[0] * img.shape[1]
    max_area = 0.05 * img.shape[0] * img.shape[1]
    area = cv2.contourArea(contour)
    return min_area < area < max_area

def segment_logo(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 0    # Brightness control (0-100)
    contrasted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # Then apply thresholding
    _, edges = cv2.threshold(contrasted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        if is_logo_candidate(contour, img):
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [contour], 255)
            segmented_image = np.where(mask[..., None] == 255, img, 255)

            # Optionally, save each segmented image
            image_name = image_path.split('/')[-1].split('.')[0]
            image_file_type = image_path.split('/')[-1].split('.')[1]
            cv2.imwrite(f'./segment/{image_name}_segmented_{i}.{image_file_type}', segmented_image)
            # If you want to plot them, you can add Matplotlib code here

# Example usage
imagePath = ['handbag8.jpeg']
for path in imagePath:
    segment_logo(path)
