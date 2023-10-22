import numpy as np
import cv2
import os

def sobel_filter(img):
    mask_n = 3
    # Define the 3x3 Sobel masks
    mask_Gx = np.array([[-1,  0, 1],
                        [-2,  0, 2],
                        [-1,  0, 1]])
    
    mask_Gy = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    
    filtered_img_Gx = np.zeros_like(img, dtype=np.float64)
    filtered_img_Gy = np.zeros_like(img, dtype=np.float64)
    
    m, n = img.shape
    offset = mask_n // 2
    for i in range(offset, m - offset): 
        for j in range(offset, n - offset): 
            mask_area = img[i - offset:i + offset + 1, j - offset:j + offset + 1]
            filtered_img_Gx[i, j] = np.sum(mask_area * mask_Gx)
            filtered_img_Gy[i, j] = np.sum(mask_area * mask_Gy)

    # # Show the filtered images using OpenCV
    # cv2.imshow('Filtered Gx', filtered_img_Gx.astype(np.uint8))
    # cv2.imshow('Filtered Gy', filtered_img_Gy.astype(np.uint8))
    
    # # Wait for a key press and then close all OpenCV windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return filtered_img_Gx, filtered_img_Gy

def enhance_with_sobel(img):
    # Compute the Sobel filtered images
    filtered_Gx, filtered_Gy = sobel_filter(img)
    
    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(filtered_Gx**2 + filtered_Gy**2)
    
    # Normalize the gradient magnitude to enhance it
    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude) * 255
    
    # Sharpen the image using the unsharp masking technique
    sharpened_img = img + gradient_magnitude
    sharpened_img = np.clip(sharpened_img, 0, 255)  # Ensure pixel values are in [0,255]
    
    return sharpened_img.astype(np.uint8)


img = cv2.imread("blurred_image.jpg", cv2.IMREAD_GRAYSCALE)

output_folder = "./sharpening_filters"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

sobel_filtered = enhance_with_sobel(img)
cv2.imwrite(os.path.join(output_folder, "sobel_filtered.jpg"), sobel_filtered)


# Example usage:
# Assuming you have an image 'img' loaded as a 2D numpy array (grayscale)
# sharpened = gradient_sharpening(img, mask_n=3)
