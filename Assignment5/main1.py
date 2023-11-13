import numpy as np
import cv2
import os

image = cv2.imread("fruit.jpg")

complementary_image = 255 - image

cv2.imshow('Original Image', image)
cv2.imshow('Complementary Image', complementary_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

output_folder = "./complementary"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
cv2.imwrite(os.path.join(output_folder,"complementary_fruit.jpg"), complementary_image)
