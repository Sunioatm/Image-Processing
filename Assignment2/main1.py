import os
import cv2
import numpy as np

def gamma_correction(image, gamma):
    enhanced_image = np.power(image / 255.0, gamma)
    enhanced_image = np.uint8(255 * enhanced_image)
    return enhanced_image

img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)

output_folder = "./gammaCorrection"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

gammaList = [x/10 for x in range(1, 10)] + [x for x in range(1, 11)]

for i in gammaList:
    tempImg = gamma_correction(img, i)
    output_path = os.path.join(output_folder, f"gamma={i}.jpg")
    cv2.imwrite(output_path, tempImg)
    