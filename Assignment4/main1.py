import numpy as np
import cv2
import os

def notch_filter(rows, cols, r, filter_type="low-pass"):
    center_x, center_y = rows // 2, cols // 2
    filter = np.zeros((rows, cols), np.float32)

    for x in range(rows):
        for y in range(cols):
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if filter_type == "low-pass":
                if distance <= r:
                    filter[x, y] = 1
            elif filter_type == "high-pass":
                if distance > r:
                    filter[x, y] = 1

    return filter

images = ["flower1.jpg", "fruit.jpg"]
for each in images:
    img = cv2.imread(each, cv2.IMREAD_GRAYSCALE)

    # Discrete Fourier Transform
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Get magnitude spectrum
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))

    # To visualize better, normalize the magnitude spectrum to range 0-255
    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    # Create the filter masks

    r_values = [10, 50, 100]
    for r in r_values:
        low_pass_filter = notch_filter(img.shape[0], img.shape[1], r, "low-pass")
        high_pass_filter = notch_filter(img.shape[0], img.shape[1], r, "high-pass")

        # Apply the filters
        low_pass = dft_shifted * low_pass_filter[:, :, np.newaxis]
        high_pass = dft_shifted * high_pass_filter[:, :, np.newaxis]

    # Inverse DFT for Low-Pass filtered image
        idft_low_pass = np.fft.ifftshift(low_pass)
        img_reconstructed_low = cv2.idft(idft_low_pass, flags=cv2.DFT_REAL_OUTPUT)
        img_reconstructed_low = cv2.normalize(img_reconstructed_low, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Inverse DFT for High-Pass filtered image
        idft_high_pass = np.fft.ifftshift(high_pass)
        img_reconstructed_high = cv2.idft(idft_high_pass, flags=cv2.DFT_REAL_OUTPUT)
        img_reconstructed_high = cv2.normalize(img_reconstructed_high, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Show the reconstructed images
        # cv2.imshow(f"Low-Pass Filtered r={r}", img_reconstructed_low)
        # cv2.imshow(f"High-Pass Filtered r={r}", img_reconstructed_high)

        output_folder = "./notch_filters"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cv2.imwrite(os.path.join(output_folder, f"{each} low pass r = {r}.jpg"), img_reconstructed_low)
        cv2.imwrite(os.path.join(output_folder, f"{each} high pass r = {r}.jpg"), img_reconstructed_high)

    # cv2.imshow("Original Image", img)
    # cv2.imshow("Magnitude Spectrum", magnitude_spectrum_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
