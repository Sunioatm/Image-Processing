import cv2
import numpy as np
import os

def enhance_image(image):
    L = 256
    enhanced_image = np.copy(image)

    h, w = image.shape

    for y in range(h):
        for x in range(w):
            if 0 <= enhanced_image[y, x] < L/3:
                enhanced_image[y, x] = int(5 * enhanced_image[y, x] / 6)
            elif L/3 <= enhanced_image[y, x] < 2 * L/3:
                slope = ((L/6) - (5 * L/6)) / ((2 * L/3) - (L/3))
                enhanced_image[y, x] = int((5 * L/6) + (slope * (enhanced_image[y, x] - (L/3))))
            else:
                enhanced_image[y, x] = int(enhanced_image[y, x] / 6)

    return enhanced_image

def main():
    asset_path = './asset'
    pictures = ["flower.jpg", "traffic.jpg", "tram.jpg"]
    for picture in pictures:
        image_path = os.path.join(asset_path, picture)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Could not open or find the image.")
            return

        enhanced_image = enhance_image(image)

        cv2.imshow("enhanced_image", enhanced_image)

        # pic_name = picture.split(".")[0]
        # output_dir = f"./enhanced/{pic_name}"
        # os.makedirs(output_dir, exist_ok=True)
        # cv2.imwrite(os.path.join(output_dir, f'{pic_name}_enhanced.jpg'), enhanced_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
