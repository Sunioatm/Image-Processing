import cv2
img = cv2.imread("blurred_image.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("original_gray.jpg",img)