import os
import cv2
import numpy as np

import matplotlib.pyplot as plt

def gaussian_notch_reject(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    d2_from_center = (x - ccol)**2 + (y - crow)**2
    d2_from_d0_above = (x - ccol)**2 + (y - (crow-d0))**2
    d2_from_d0_below = (x - ccol)**2 + (y - (crow+d0))**2

    # Gaussian masks for the two dots
    mask_above = np.exp(-d2_from_d0_above / (2*width**2))
    mask_below = np.exp(-d2_from_d0_below / (2*width**2))
    
    # Final combined mask (1 - mask) to reject the specific frequencies
    mask_gaussian = 1 - (mask_above + mask_below)

    # Extend the shape for real and imaginary parts
    mask_gaussian = mask_gaussian[:, :, np.newaxis]

    return mask_gaussian

img = cv2.imread('noisy_flower1_horizontal.jpg', cv2.IMREAD_GRAYSCALE)


dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

notch70 = gaussian_notch_reject(img.shape, 70, 25)
notch140 = gaussian_notch_reject(img.shape, 140, 25)

notch = notch70 * notch140
# Apply the notch filter
fshift = dft_shifted * notch

# Inverse FFT to get the image back
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift, flags=cv2.DFT_REAL_OUTPUT)
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

# Compute the magnitude of the filtered frequency domain
magnitude_values_filtered = cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])
magnitude_spectrum_filtered = 20 * np.log(magnitude_values_filtered + 1)
magnitude_spectrum_filtered_normalized = cv2.normalize(magnitude_spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Original Magnitude Spectrum', magnitude_spectrum_normalized.astype(np.uint8))
cv2.imshow('Filtered Magnitude Spectrum', magnitude_spectrum_filtered_normalized.astype(np.uint8))


output_folder = "./noisy"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cv2.imwrite(os.path.join(output_folder, f"horizontal image d0 = 70,140 width = 25.jpg"), img_back)
cv2.imwrite(os.path.join(output_folder, f"horizontal magnitude_spectrum.jpg"), magnitude_spectrum_normalized)
cv2.imwrite(os.path.join(output_folder, f"horizontal magnitude_spectrum_filtered d0 = 70,140 width = 25.jpg"), magnitude_spectrum_filtered_normalized)
