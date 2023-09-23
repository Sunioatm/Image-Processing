import os
import cv2
import numpy as np

def global_histogram_equalization(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    # cumulative distribution function
    cdf = hist.cumsum()
    # print("cdf : \n",cdf)
    cdf_normalized = ((cdf - cdf.min())* 255) / cdf[-1] #Normalize to range [0, 255]
    # print("cdf normalizaed : \n",cdf_normalized)
    img_equalized = cdf_normalized[img] # For each pixel in the img, find its corresponding equalized value in cdf_normalized.
    return img_equalized.astype(np.uint8)

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
imgGlobalEqualized = global_histogram_equalization(img)

output_folder = "./globalEqualization"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
cv2.imwrite(os.path.join(output_folder,f"global equalization.jpg"), imgGlobalEqualized)