import os
import cv2
import numpy as np

def local_histogram_equalization(img, kernel_size=5, k0=0.4, k1=0.02, k2=0.4):
    half_size = kernel_size // 2
    # For edge of picture
    padded_img = np.pad(img, ((half_size, half_size), (half_size, half_size)), mode='reflect')
    img_equalized = np.copy(img)

    global_mean = np.mean(img)
    global_deviation = np.std(img)

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

image_path = 'image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

setk0 = [i/10 for i in range(1,11)]
setk1 = [i/100 for i in range(1,11)]
setk2 = [i/10 for i in range(1,11)]

# # 3x3
for k0 in setk0:
    for k1 in setk1:
        for k2 in setk2:
            tempImg = local_histogram_equalization(image,3,k0,k1,k2)
            cv2.imwrite(os.path.join("localEqualization","3x3", f'3x3 k0={k0},k1={k1},k2={k2}.jpg'), tempImg)

# # 7x7
for k0 in setk0:
    for k1 in setk1:
        for k2 in setk2:
            tempImg = local_histogram_equalization(image,7,k0,k1,k2)
            cv2.imwrite(os.path.join("localEqualization","7x7", f'7x7 k0={k0},k1={k1},k2={k2}.jpg'), tempImg)
# # 11x11
for k0 in setk0:
    for k1 in setk1:
        for k2 in setk2:
            tempImg = local_histogram_equalization(image,11,k0,k1,k2)
            cv2.imwrite(os.path.join("localEqualization","11x11", f'11x11 k0={k0},k1={k1},k2={k2}.jpg'), tempImg)


