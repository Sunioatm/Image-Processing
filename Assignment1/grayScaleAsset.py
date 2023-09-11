import os
import cv2
import numpy as np

def map_to_gray_levels(image, levels):
    step = 256 // (levels)
    new_image = (image // step) * step
    return new_image.astype(np.uint8)

def main():
    asset_path = './asset'
    pictures = ["cartoon.jpg", "flower.jpg", "fractal.jpeg", "fruit.jpg", "scenery1.jpg", "scenery2.jpg", "traffic.jpg", "tram.jpg"]

    for picture in pictures:
        image_path = os.path.join(asset_path, picture)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Could not open or find the image.")
            return

        # gray256 = map_to_gray_levels(image, 256)

        pic_name = picture.split(".")[0]
        # output_dir = f"./gray_asset/{pic_name}"
        # os.makedirs(output_dir, exist_ok=True)


        cv2.imwrite(f'./gray_asset/{pic_name}_gray.jpg', image)


if __name__ == '__main__':
    main()
