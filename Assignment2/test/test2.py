import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def global_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    return equ

def local_histogram_equalization(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ksize, ksize))
    clahe_img = clahe.apply(gray)
    return clahe_img

image_path = 'test.jpg'
image = cv2.imread(image_path, 1)

gamma_corrected = gamma_correction(image, gamma=1.5)

global_hist_equalized = global_histogram_equalization(image)

local_hist_equalized_3x3 = local_histogram_equalization(image, 3)
local_hist_equalized_7x7 = local_histogram_equalization(image, 7)
local_hist_equalized_11x11 = local_histogram_equalization(image, 11)

cv2.imshow('Original', image)
cv2.imshow('Gamma Correction', gamma_corrected)
cv2.imshow('Global Histogram Equalization', global_hist_equalized)
cv2.imshow('Local Histogram Equalization 3x3', local_hist_equalized_3x3)
cv2.imshow('Local Histogram Equalization 7x7', local_hist_equalized_7x7)
cv2.imshow('Local Histogram Equalization 11x11', local_hist_equalized_11x11)

cv2.waitKey(0)
cv2.destroyAllWindows()
