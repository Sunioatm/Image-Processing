import cv2
import numpy as np

def global_histogram_equalization(img):
    hist, _ = np.histogram(img.flatten(), 256, [0, 256])
    # cumulative distribution function
    cdf = hist.cumsum()
    print("cdf : \n",cdf)
    cdf_normalized = (cdf * 255) / cdf[-1]  # Normalize to range [0, 255]
    print("cdf normalizaed : \n",cdf_normalized)
    img_equalized = cdf_normalized[img]
    return img_equalized.astype(np.uint8)

# def local_histogram_equalization(img, ksize):
#     half = ksize // 2
#     img_padded = np.pad(img, ((half, half), (half, half)), 'reflect')
#     result = np.zeros_like(img)
    
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             local_region = img_padded[i:i+ksize, j:j+ksize]
#             result[i, j] = global_histogram_equalization(local_region)[half, half]
            
#     return result

image_path = 'test.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

global_hist_equalized = global_histogram_equalization(image)
# local_hist_equalized_3x3 = local_histogram_equalization(image, 3)
# local_hist_equalized_7x7 = local_histogram_equalization(image, 7)
# local_hist_equalized_11x11 = local_histogram_equalization(image, 11)


cv2.imshow('Original', image)
cv2.imshow('Global Histogram Equalization', global_hist_equalized)
# cv2.imshow('Local Histogram Equalization 3x3', local_hist_equalized_3x3)
# cv2.imshow('Local Histogram Equalization 7x7', local_hist_equalized_7x7)
# cv2.imshow('Local Histogram Equalization 11x11', local_hist_equalized_11x11)


cv2.waitKey(0)
cv2.destroyAllWindows()