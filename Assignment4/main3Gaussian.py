import numpy as np
import cv2

def gaussian_notch_reject(shape, d0, width):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Calculate distance from the center
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x*x + y*y)

    # Apply Gaussian formula
    mask = np.exp(-(d**2) / (2*(d0**2)))
    r = width // 2
    mask[crow - r:crow + r, :] = 1

    return mask

if __name__ == '__main__':
    img = cv2.imread('noisy_flower1_horizontal.jpg', cv2.IMREAD_GRAYSCALE)
    mask = gaussian_notch_reject(img.shape, d0=70, width=50)
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
