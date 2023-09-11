import cv2
import os
import numpy as np

def enhance_image(image, c, gamma):
    enhanced_image = np.power(image / 255.0, gamma)
    enhanced_image = np.uint8(255 * c * enhanced_image)
    return enhanced_image

def main():
    asset_path = './asset'
    pictures = ["cartoon.jpg", "scenery1.jpg", "scenery2.jpg"]
    for picture in pictures:
        image_path = os.path.join(asset_path, picture)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Could not open or find the image.")
            return

        c_values = [0.5, 1, 2]
        gamma_values = [0.4, 2.5]

        for c in c_values:
            for gamma in gamma_values:
                enhanced_image = enhance_image(image, c, gamma)
                cv2.imshow(f"enhanced_image c : {c} gamma : {gamma}", enhanced_image)

                pic_name = picture.split(".")[0]
                output_dir = f"./enhanced_powerlaw/{pic_name}"
                os.makedirs(output_dir, exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f'{pic_name}_c{c}_gamma{gamma}.jpg'), enhanced_image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
