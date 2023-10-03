import cv2
import numpy as np
import matplotlib as plt
# from scipy.ndimage import uniform_filter

def global_histogram_equalization(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    # cumulative distribution function
    cdf = hist.cumsum()
    # print("cdf : \n",cdf)
    cdf_normalized = ((cdf - cdf.min())* 255) / cdf[-1] #Normalize to range [0, 255]
    # print("cdf normalizaed : \n",cdf_normalized)
    img_equalized = cdf_normalized[img] # For each pixel in the img, find its corresponding equalized value in cdf_normalized.
    return img_equalized.astype(np.uint8)

def local_histogram_equalization(img, kernel_size=3, k0=0.4, k1=0.02, k2=0.4):
    half_size = kernel_size//2
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size)), mode='reflect')
    img_equalized = np.copy(img)

    global_mean = np.mean(img)
    global_deviation = np.std(img)
    
    # k0 = 0.3
    # k1 = 0.05
    # k2 = 0.4
    
    # setOfk = [(), (), (), (), (), (), (), (), (), ()]
    
    for i in range(half_size, img.shape[0] - half_size):
        for j in range(half_size, img.shape[1] - half_size):
            local_region = padded_img[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            local_mean = np.mean(local_region)
            local_deviation = np.std(local_region)

            # print("global mean",global_mean)
            # print("local mean", local_mean)

            # print("global sd",global_deviation)
            # print("local sd",local_deviation)

            # k0 = 0.35
            # k1 = 0.02
            # k2 = 0.03

            if local_mean < k0 * global_mean and k1 * global_deviation <= local_deviation <= k2 * global_deviation:
                hist, _ = np.histogram(local_region.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                img_equalized[i, j] = cdf_normalized[img[i, j]]

    return img_equalized

image_path = 'test.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

global_hist_equalized = global_histogram_equalization(image)
local_hist_equalized_3x3 = local_histogram_equalization(image, 3, 0.125, 0.01, 0.125)
# local_hist_equalized_7x7 = local_histogram_equalization(image, 7)
# local_hist_equalized_11x11 = local_histogram_equalization(image, 11)


cv2.imshow('Original', image)
cv2.imshow('Global Histogram Equalization', global_hist_equalized)
cv2.imshow('Local Histogram Equalization 3x3', local_hist_equalized_3x3)
# cv2.imshow('Local Histogram Equalization 7x7', local_hist_equalized_7x7)
# cv2.imshow('Local Histogram Equalization 11x11', local_hist_equalized_11x11)


cv2.waitKey(0)
cv2.destroyAllWindows()

