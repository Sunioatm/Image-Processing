import cv2
import numpy as np
from scipy.ndimage import uniform_filter

def global_histogram_equalization(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    # cumulative distribution function
    cdf = hist.cumsum()
    # print("cdf : \n",cdf)
    cdf_normalized = ((cdf - cdf.min())* 255) / cdf[-1] #Normalize to range [0, 255]
    # print("cdf normalizaed : \n",cdf_normalized)
    img_equalized = cdf_normalized[img] # For each pixel in the img, find its corresponding equalized value in cdf_normalized.
    return img_equalized.astype(np.uint8)

def local_histogram_equalization(img, kernel_size=3):
    # Pad for edge of picture
    half_size = kernel_size//2
    # padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size)), mode='reflect')
    img_equalized = np.copy(img)

    # img_equalized = np.zeros_like(img, dtype=np.uint8)
    
    global_mean = np.mean(img)
    global_deviation = np.std(img)
    # print(global_mean)    
    # global_hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    # global_bin_midpoints = (bins[:-1] + bins[1:]) / 2
    # global_mean = np.sum(global_bin_midpoints * global_hist) / np.sum(global_hist)
    # print(global_mean)
    
    for i in range(half_size, img.shape[0]-half_size + 1):
        for j in range(half_size, img.shape[1]-half_size + 1):

            # Extract local region
            # local_region = padded_img[i-half_size:i+half_size+1, j-half_size:j+half_size+1]

            # print(f"img : {img}")    

            local_region = img[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            # sigma = np.std(local_region)
            # if sigma <= global_deviation:
            
            # Equalize the local region
            # hist = represents the number of pixels in local_region

            hist, bins = np.histogram(local_region.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            
            if cdf[-1] == 0:
                continue
            
            cdf_normalized = (cdf - cdf.min()) * 255 / cdf[-1]
            # cdf_normalized = (cdf) * 255 / cdf[-1]
            # print(f"cdf : {cdf}")
            # print(f"cdf normalized : {cdf_normalized}")
            central_pixel = local_region[half_size, half_size]
            mean = np.mean(local_region)
            
            if mean < global_mean:
                img_equalized[i, j] = cdf_normalized[central_pixel]
                # img_equalized[i, j] = central_pixel

            else :
                pass

    return img_equalized

image_path = 'test.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

global_hist_equalized = global_histogram_equalization(image)
local_hist_equalized_3x3 = local_histogram_equalization(image, 3)
# local_hist_equalized_7x7 = local_histogram_equalization(image, 7)
# local_hist_equalized_11x11 = local_histogram_equalization(image, 11)


cv2.imshow('Original', image)
cv2.imshow('Global Histogram Equalization', global_hist_equalized)
cv2.imshow('Local Histogram Equalization 3x3', local_hist_equalized_3x3)
# cv2.imshow('Local Histogram Equalization 7x7', local_hist_equalized_7x7)
# cv2.imshow('Local Histogram Equalization 11x11', local_hist_equalized_11x11)


cv2.waitKey(0)
cv2.destroyAllWindows()