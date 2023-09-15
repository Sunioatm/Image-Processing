import numpy as np

def local_histogram_equalization(image, ksize):
    pad_size = ksize // 2
    padded_img = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size)), 'constant')
    equalized_img = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            neighborhood = padded_img[i:i+ksize, j:j+ksize]
            equalized_img[i, j] = global_histogram_equalization(neighborhood)[pad_size, pad_size]
    
    return equalized_img

def global_histogram_equalization(image):
    # Compute the histogram of the image
    hist, _ = np.histogram(image.flatten(), 256, [0,256])
    
    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Normalize the CDF
    cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
    cdf_mapped = np.uint8(cdf_normalized)
    
    # Use the CDF to map the original pixel values to the equalized values
    equalized_img = cdf_mapped[image]
    
    return equalized_img