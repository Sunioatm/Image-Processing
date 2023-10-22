import os
import cv2
import numpy as np

def averaging_filter(img, mask_n=3):
    mask = np.ones([mask_n,mask_n], dtype=np.float32)
    mask = mask / (mask_n**2)
    filtered_img = np.zeros_like(img)
    m,n = img.shape
    offset = mask_n // 2
    for i in range(offset, m-offset): 
        for j in range(offset, n-offset): 
            mask_area = img[i-offset:i+offset+1, j-offset:j+offset+1]
            filtered_img[i, j]= np.sum(mask_area*mask)

    return filtered_img

def median_filter(img, mask_n=3):
    mask = np.ones([mask_n,mask_n], dtype=np.float32)
    filtered_img = np.zeros_like(img)
    m,n = img.shape
    offset = mask_n // 2
    for i in range(offset, m-offset): 
        for j in range(offset, n-offset): 
            mask_area = img[i-offset:i+offset+1, j-offset:j+offset+1]
            mask_flatten = (mask_area*mask).flatten()
            filtered_img[i, j]= np.median(mask_flatten)
    return filtered_img

img1 = cv2.imread("noisy_img1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("noisy_img2.jpg", cv2.IMREAD_GRAYSCALE)

output_folder = "./smoothing_filters"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

avg_filtered_1  = averaging_filter(img1, 3)
cv2.imwrite(os.path.join(output_folder, "noisy1_avg_filtered.jpg"), avg_filtered_1)

med_filtered_1 = median_filter(img1,3)
cv2.imwrite(os.path.join(output_folder, "noisy1_med_filtered.jpg"), med_filtered_1)


avg_filtered_2  = averaging_filter(img2, 3)
cv2.imwrite(os.path.join(output_folder, "noisy2_avg_filtered.jpg"), avg_filtered_2)

med_filtered_2 = median_filter(img2,3)
cv2.imwrite(os.path.join(output_folder, "noisy2_med_filtered.jpg"), med_filtered_2)