import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def global_histogram_equalization(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    # cumulative distribution function
    cdf = hist.cumsum()
    # print("cdf : \n",cdf)
    cdf_normalized = ((cdf - cdf.min())* 255) / cdf[-1] #Normalize to range [0, 255]
    # print("cdf normalizaed : \n",cdf_normalized)
    img_equalized = cdf_normalized[img] # For each pixel in the img, find its corresponding equalized value in cdf_normalized.
    return img_equalized.astype(np.uint8)

def local_histogram_equalization(img, kernel_size=5, k0=0.4, k1=0.02, k2=0.4):
    half_size = kernel_size // 2
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size)), mode='reflect')
    img_equalized = np.copy(img)

    global_mean = np.mean(img)
    global_deviation = np.std(img)
    global_variance = np.var(img)

    # k0 = 0.35
    # k1 = 0.03
    # k2 = 0.5

    for i in range(half_size, img.shape[0] - half_size):
        for j in range(half_size, img.shape[1] - half_size):
            local_region = padded_img[i-half_size:i+half_size+1, j-half_size:j+half_size+1]
            local_mean = np.mean(local_region)
            local_deviation = np.std(local_region)

            if local_mean < k0 * global_mean and k1 * global_deviation <= local_deviation <= k2 * global_deviation:
                hist, _ = np.histogram(local_region.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                img_equalized[i, j] = cdf_normalized[img[i, j]]

    return img_equalized

image_path = 'test.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

setk0 = [i/10 for i in range(1,11)]
setk1 = [i/100 for i in range(1,11)]
setk2 = [i/10 for i in range(1,11)]

# # 3x3
# for k0 in setk0:
#     for k1 in setk1:
#         for k2 in setk2:
#             tempImg = local_histogram_equalization(image,3,k0,k1,k2)
#             cv2.imwrite(os.path.join("localEnhanecmentOutput","3x3", f'3x3 k0={k0},k1={k1},k2={k2}.jpg'), tempImg)

# # 7x7
# for k0 in setk0:
#     for k1 in setk1:
#         for k2 in setk2:
#             tempImg = local_histogram_equalization(image,7,k0,k1,k2)
#             cv2.imwrite(os.path.join("localEnhanecmentOutput","7x7", f'7x7 k0={k0},k1={k1},k2={k2}.jpg'), tempImg)
# # 11x11
# for k0 in setk0:
#     for k1 in setk1:
#         for k2 in setk2:
#             tempImg = local_histogram_equalization(image,11,k0,k1,k2)
#             cv2.imwrite(os.path.join("localEnhanecmentOutput","11x11", f'11x11 k0={k0},k1={k1},k2={k2}.jpg'), tempImg)

# k0 = 0.35
# k1= 0.03
# k2 = 0.5
# tempImg = local_histogram_equalization(image,3,k0,k1,k2)
# cv2.imshow("test",tempImg)
# cv2.imwrite(os.path.join("testOutput","3x3", f'k0={k0},k1={k1},k2={k2}.jpg'), tempImg)

# global_hist_equalized = global_histogram_equalization(image)
# local_hist_equalized_3x3 = local_histogram_equalization(image, 3)
# local_hist_equalized_7x7 = local_histogram_equalization(image, 7)
# local_hist_equalized_11x11 = local_histogram_equalization(image, 11)


# cv2.imshow('Original', image)
# cv2.imshow('Global Histogram Equalization', global_hist_equalized)
# cv2.imshow('Local Histogram Equalization 3x3', local_hist_equalized_3x3)
# cv2.imshow('Local Histogram Equalization 7x7', local_hist_equalized_7x7)
# cv2.imshow('Local Histogram Equalization 11x11', local_hist_equalized_11x11)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

