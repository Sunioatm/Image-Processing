import cv2
import numpy as np
import os

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

images = ["flower1.jpg", "fruit.jpg"]
for each in images:
    # Load the image in grayscale
    img = cv2.imread(each, cv2.IMREAD_GRAYSCALE)

    # Apply FFT and shift the zero-frequency component to the center
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

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

        # High-pass filter
        hpf = gaussian_hpf(img.shape[0], img.shape[1], D0)
        filtered_hpf = dft_shifted * hpf
        idft_hpf = np.fft.ifftshift(filtered_hpf)
        img_reconstructed_hpf = cv2.idft(idft_hpf, flags=cv2.DFT_REAL_OUTPUT)
        img_display_hpf = cv2.normalize(img_reconstructed_hpf, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #     cv2.imshow(f'Gaussian HPF D0={D0}', img_display_hpf)

        output_folder = "./gaussian_filters"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cv2.imwrite(os.path.join(output_folder, f"{each} low pass d0 = {D0}.jpg"), img_display_lpf)
        cv2.imwrite(os.path.join(output_folder, f"{each} high pass d0 = {D0}.jpg"), img_display_hpf)


    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
