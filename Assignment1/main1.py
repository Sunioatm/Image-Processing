import os
import cv2
import numpy as np

def map_to_gray_levels(image, levels):
    step = 256 // (levels)
    new_image = (image // step) * step
    return new_image.astype(np.uint8)

def main():
    asset_path = './asset'
    pictures = ["flower.jpg", "fractal.jpeg", "fruit.jpg"]

    for picture in pictures:
        image_path = os.path.join(asset_path, picture)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Could not open or find the image.")
            return

        gray8 = map_to_gray_levels(image, 8)
        gray64 = map_to_gray_levels(image, 64)
        gray128 = map_to_gray_levels(image, 128)
        gray256 = map_to_gray_levels(image, 256)

        pic_name = picture.split(".")[0]
        output_dir = f"./gray/{pic_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Display images
        cv2.imshow(f'{pic_name}_original', image)
        cv2.imshow(f'{pic_name}_gray8', gray8)
        cv2.imshow(f'{pic_name}_gray64', gray64)
        cv2.imshow(f'{pic_name}_gray128', gray128)
        cv2.imshow(f'{pic_name}_gray256', gray256)

        # Save images
        cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray8.jpg'), gray8)
        cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray64.jpg'), gray64)
        cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray128.jpg'), gray128)
        cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray256.jpg'), gray256)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
