import cv2
import numpy as np
import matplotlib.pyplot as plt
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

            # print("i,j",i,j)
            # print("global sd",global_deviation)
            # print("local sd",local_deviation)


            if local_mean < k0 * global_mean and k1 * global_deviation <= local_deviation <= k2 * global_deviation:
                hist, _ = np.histogram(local_region.flatten(), 256, [0, 256])
                cdf = hist.cumsum()
                cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
                img_equalized[i, j] = cdf_normalized[img[i, j]]

    # Plotting
    # plt.figure(figsize=(12, 8))

    # # Original Image Histogram
    # plt.subplot(2, 2, 1)
    # plt.hist(img.ravel(), 256, [0, 256], color='blue', alpha=0.7)
    # plt.axvline(global_mean, color='red', linestyle='--')
    # plt.axvline(global_mean + global_deviation, color='green', linestyle='--')
    # plt.axvline(global_mean - global_deviation, color='green', linestyle='--')
    # plt.annotate(f'Mean: {global_mean:.2f}', (global_mean, 5), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.annotate(f'SD: +/- {global_deviation:.2f}', (global_mean + global_deviation, 5), textcoords="offset points", xytext=(50,10), ha='center')
    # plt.title('Original Image Histogram')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')

    # # Equalized Image Histogram
    # equalized_mean = np.mean(img_equalized)
    # equalized_deviation = np.std(img_equalized)

    # plt.subplot(2, 2, 2)
    # plt.hist(img_equalized.ravel(), 256, [0, 256], color='green', alpha=0.7)
    # plt.axvline(equalized_mean, color='red', linestyle='--')
    # plt.axvline(equalized_mean + equalized_deviation, color='purple', linestyle='--')
    # plt.axvline(equalized_mean - equalized_deviation, color='purple', linestyle='--')
    # plt.annotate(f'Mean: {equalized_mean:.2f}', (equalized_mean, 5), textcoords="offset points", xytext=(0,10), ha='center')
    # plt.annotate(f'SD: +/- {equalized_deviation:.2f}', (equalized_mean + equalized_deviation, 5), textcoords="offset points", xytext=(50,10), ha='center')
    # plt.title('Equalized Image Histogram')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')

    # # Original Image CDF
    # plt.subplot(2, 2, 3)
    # hist_original, _ = np.histogram(img.ravel(), 256, [0, 256])
    # cdf_original = hist_original.cumsum()
    # plt.plot(cdf_original, color='blue', alpha=0.7)
    # plt.title('Original Image CDF')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Cumulative Frequency')

    # # Equalized Image CDF
    # plt.subplot(2, 2, 4)
    # hist_equalized, _ = np.histogram(img_equalized.ravel(), 256, [0, 256])
    # cdf_equalized = hist_equalized.cumsum()
    # plt.plot(cdf_equalized, color='green', alpha=0.7)
    # plt.title('Equalized Image CDF')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Cumulative Frequency')

    # plt.tight_layout()
    # plt.show()

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

