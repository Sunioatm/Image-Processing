import cv2
import numpy as np

def imgToDftShifted(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)
    return dft_shifted


def dftShiftedToImg(dftImg):
    idft = np.fft.ifftshift(dftImg)
    img = cv2.idft(idft, flags=cv2.DFT_REAL_OUTPUT)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img
    
def magnitudeSpectrum(dft_shifted):
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1]))
    magnitude_spectrum_normalized = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude_spectrum_normalized