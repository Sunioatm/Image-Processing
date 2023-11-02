import cv2
import numpy as np

import matplotlib.pyplot as plt

def onclick(event):
    center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
    d0 = ((event.xdata - center_x)**2 + (event.ydata - center_y)**2)**0.5
    print(f'd0 = {d0}')

def notch_reject_filter(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)
    r = width // 2
    
    for i in range(-r, r+1):
        # mask[crow - d0 + i, ccol - r : ccol + r + 1] = 0
        # mask[crow + d0 + i, ccol - r : ccol + r + 1] = 0
        mask[crow - r : crow + r + 1, ccol - d0 + i] = 0
        mask[crow - r : crow + r + 1, ccol + d0 + i] = 0

    return mask



# def gaussian_notch_reject(shape, d0=30, width=10):
#     rows, cols = shape
#     crow, ccol = rows // 2, cols // 2

#     y, x = np.ogrid[:rows, :cols]
#     mask = np.ones((rows, cols, 2), np.float32)
#     d2 = (x - ccol)**2 + (y - crow)**2

#     mask_gaussian = 1 - np.exp(-d2 / (2*d0**2))
    
#     # Applying rectangular regions to nullify the undesired frequencies
#     mask[crow - width//2 : crow + width//2, ccol - width//2 : ccol + width//2] = 1

#     return mask_gaussian[:, :, np.newaxis]

def gaussian_notch_reject(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    d2_from_center = (x - ccol)**2 + (y - crow)**2
    # d2_from_d0_above = (x - ccol)**2 + (y - (crow-d0))**2
    # d2_from_d0_below = (x - ccol)**2 + (y - (crow+d0))**2

    # # Gaussian masks for the two dots
    # mask_above = np.exp(-d2_from_d0_above / (2*width**2))
    # mask_below = np.exp(-d2_from_d0_below / (2*width**2))
    d2_from_d0_right = (x - (ccol-d0))**2 + (y - (crow))**2
    d2_from_d0_left = (x - (ccol+d0))**2 + (y - (crow))**2

    # Gaussian masks for the two dots
    mask_above = np.exp(-d2_from_d0_right / (2*width**2))
    mask_below = np.exp(-d2_from_d0_left / (2*width**2))
    
    # Final combined mask (1 - mask) to reject the specific frequencies
    mask_gaussian = 1 - (mask_above + mask_below)

    # Extend the shape for real and imaginary parts
    mask_gaussian = mask_gaussian[:, :, np.newaxis]

    return mask_gaussian

img = cv2.imread('noisy_flower1_vertical.jpg', cv2.IMREAD_GRAYSCALE)

# d0 = [10,20,30,40,50]
# width = [5,10,15,20]
# for i in d0:
#     for j in width:
        
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Create the notch filter
# notch1 = notch_reject_filter(img.shape, 95, 20)
# notch2 = notch_reject_filter(img.shape, 190, 20)

notch1 = gaussian_notch_reject(img.shape,95, 25)
notch2 = gaussian_notch_reject(img.shape, 190, 25)

notch = notch1 * notch2

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


# fig, ax = plt.subplots()
# ax.imshow(magnitude_spectrum_normalized, cmap='gray', extent=[-img.shape[1]//2, img.shape[1]//2, -img.shape[0]//2, img.shape[0]//2])
# ax.set_xlabel("Frequency (u)")
# ax.set_ylabel("Frequency (v)")
# cid = fig.canvas.mpl_connect('button_press_event', onclick)

# plt.show()

cv2.imshow('Original', img)
# cv2.imshow(f'Processed d0 = {i} width = {j}', img_back)
cv2.imshow('Processed', img_back)

cv2.waitKey(0)
cv2.destroyAllWindows()
