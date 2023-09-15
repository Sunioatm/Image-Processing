import cv2

def global_histogram_equalization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(gray)
    return equ

def local_histogram_equalization(image, ksize):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(ksize, ksize))
    clahe_img = clahe.apply(gray)
    return clahe_img

