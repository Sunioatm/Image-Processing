import os
import cv2
import numpy as np

def main():
    asset_path = './asset'
    pictures = ["flower.jpg", "fractal.jpeg", "fruit.jpg"]

    for picture in pictures:
        image_path = os.path.join(asset_path, picture)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        print(image.size)
        print(image.shape[1])

        if image is None:
            print("Could not open or find the image.")
            return

        # Define the division factors
        factors = [8, 64, 128, 256]

        pic_name = picture.split(".")[0]
        output_dir = f"./testGray/{pic_name}"
        os.makedirs(output_dir, exist_ok=True)

        for factor in factors:
            # Create a copy of the image
            gray = image.copy()
            
            # Divide each pixel value by the factor and convert to uint8
            gray = (gray // factor).astype(np.uint8)

            # Save the processed image
            cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray{factor}.jpg'), gray)

if __name__ == '__main__':
    main()
