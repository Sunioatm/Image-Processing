import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to create Gaussian Low-Pass Filter
def gaussian_lpf(rows, cols, D0):
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols, 2), np.float32)
    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            filter[x, y] = np.exp(-(distance ** 2) / (2 * (D0 ** 2)))
    return filter

# Function to create Gaussian High-Pass Filter
def gaussian_hpf(rows, cols, D0):
    return 1 - gaussian_lpf(rows, cols, D0)

def display_results(original, magnitude_before, magnitude_after, reconstructed, title):
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")

    plt.subplot(1, 4, 2)
    plt.imshow(magnitude_before, cmap='gray')
    plt.title("Magnitude Spectrum (Before)")

    plt.subplot(1, 4, 3)
    plt.imshow(magnitude_after, cmap='gray')
    plt.title("Magnitude Spectrum (After)")

    plt.subplot(1, 4, 4)
    plt.imshow(reconstructed, cmap='gray')
    plt.title(title)

    plt.tight_layout()
    plt.show()


def compute_magnitude_spectrum(dft_data):
    return 20 * np.log(np.abs(dft_data[:,:,0] + 1j*dft_data[:,:,1]) + 1)


# Load the image in grayscale
img = cv2.imread('flower1.jpg', cv2.IMREAD_GRAYSCALE)

# Apply FFT and shift the zero-frequency component to the center
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shifted[:,:,0] + 1j*dft_shifted[:,:,1]) + 1)

magnitude_spectrum_before = compute_magnitude_spectrum(dft_shifted)

# Apply Gaussian Filters
D0_values = [10, 50, 100]
for D0 in D0_values:
    # Low-pass filter
    lpf = gaussian_lpf(img.shape[0], img.shape[1], D0)
    filtered_lpf = dft_shifted * lpf
    idft_lpf = np.fft.ifftshift(filtered_lpf)
    img_reconstructed_lpf = cv2.idft(idft_lpf, flags=cv2.DFT_REAL_OUTPUT)
    img_display_lpf = cv2.normalize(img_reconstructed_lpf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # cv2.imshow(f'Gaussian LPF D0={D0}', img_display_lpf)
    display_results(img, magnitude_spectrum, img_display_lpf, f'Gaussian LPF D0={D0}')
    # High-pass filter
    hpf = gaussian_hpf(img.shape[0], img.shape[1], D0)
    filtered_hpf = dft_shifted * hpf
    idft_hpf = np.fft.ifftshift(filtered_hpf)
    img_reconstructed_hpf = cv2.idft(idft_hpf, flags=cv2.DFT_REAL_OUTPUT)
    img_display_hpf = cv2.normalize(img_reconstructed_hpf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     cv2.imshow(f'Gaussian HPF D0={D0}', img_display_hpf)
    magnitude_spectrum_after_lpf = compute_magnitude_spectrum(filtered_lpf)
    magnitude_spectrum_after_hpf = compute_magnitude_spectrum(filtered_hpf)
    
    display_results(img, magnitude_spectrum_before, magnitude_spectrum_after_lpf, img_display_lpf, f'Gaussian LPF D0={D0}')
    display_results(img, magnitude_spectrum_before, magnitude_spectrum_after_hpf, img_display_hpf, f'Gaussian HPF D0={D0}')
        
    output_folder = "./gaussian_filters"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cv2.imwrite(os.path.join(output_folder, f"low pass d0 = {D0}.jpg"), img_display_lpf)
    cv2.imwrite(os.path.join(output_folder, f"high pass d0 = {D0}.jpg"), img_display_hpf)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
