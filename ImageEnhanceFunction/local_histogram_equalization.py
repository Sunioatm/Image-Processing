import numpy as np

def local_histogram_equalization(img, kernel_size, k0, k1, k2):
    half_size = kernel_size // 2
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