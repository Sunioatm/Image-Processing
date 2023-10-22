import os
import cv2
import numpy as np


def laplacian_filter(img):
    mask_n = 3
    mask1 = np.array([[ 0,  1,  0],
                      [ 1, -4 , 1],
                      [ 0,  1,  0]])
    
    mask2 = np.array([[ 1,  1,  1],
                      [ 1, -8,  1],
                      [ 1,  1,  1]])
    
    laplacian1 = np.zeros_like(img, dtype=np.int16)
    laplacian2 = np.zeros_like(img, dtype=np.int16)
    
    m, n = img.shape
    offset = mask_n // 2
    for i in range(offset, m-offset): 
        for j in range(offset, n-offset): 
            mask_area = img[i-offset:i+offset+1, j-offset:j+offset+1]
            laplacian1[i, j] = np.sum(mask_area * mask1)
            laplacian2[i, j] = np.sum(mask_area * mask2)

    # filtered_img1 = enhance_with_laplacian(img, laplacian1)
    # filtered_img2 = enhance_with_laplacian(img, laplacian2)
    
    return laplacian1, laplacian2

def enhance_with_laplacian(img):
    laplacian1, laplacian2 = laplacian_filter(img)
    subtracted_img_1 = img.astype(np.int16) - laplacian1
    subtracted_img_2 = img.astype(np.int16) - laplacian2
    clipped_img_1 = np.clip(subtracted_img_1, 0, 255).astype(np.uint8)
    clipped_img_2 = np.clip(subtracted_img_2, 0, 255).astype(np.uint8)
    return clipped_img_1, clipped_img_2

img = cv2.imread("blurred_image.jpg", cv2.IMREAD_GRAYSCALE)

output_folder = "./sharpening_filters"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

laplacian_filtered_1, laplacian_filtered_2  = enhance_with_laplacian(img)
cv2.imwrite(os.path.join(output_folder, "laplacian_filtered_4-neighbor.jpg"), laplacian_filtered_1)
cv2.imwrite(os.path.join(output_folder, "laplacian_filtered_8-neighbor.jpg"), laplacian_filtered_2)

