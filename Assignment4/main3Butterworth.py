import numpy as np
import cv2

def butterworth_notch_reject(shape, d0, n):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    # Calculate distance from the center
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x*x + y*y)

    # Apply Butterworth formula
    mask = 1 / (1 + (d0 / d)**(2*n))
    mask[crow, :] = 1

    return mask

if __name__ == '__main__':
    img = cv2.imread('noisy_flower1_horizontal.jpg', cv2.IMREAD_GRAYSCALE)
    mask = butterworth_notch_reject(img.shape, d0=30, n=2)
    cv2.imshow('Original', img)
    cv2.imshow('Mask', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
