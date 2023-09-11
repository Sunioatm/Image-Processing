import os
import cv2
import numpy as np

def map_to_gray_levels(image, levels):
    step = 256 // (levels)
    new_image = (image // step) * step
    return new_image.astype(np.uint8)

def print_middle_values(image, window_size=3):
    # Ensure that window_size is odd to have a perfect center
    if window_size % 2 == 0:
        print("Window size should be odd!")
        return

    # Calculate starting and ending indices for rows and columns
    mid_row, mid_col = image.shape[0] // 2, image.shape[1] // 2
    half_window = window_size // 2
    
    start_row, end_row = mid_row - half_window, mid_row + half_window + 1
    start_col, end_col = mid_col - half_window, mid_col + half_window + 1

    # Extract and print the central values
    central_values = image[start_row:end_row, start_col:end_col]
    print(central_values)

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

        print("image")
        print(image)
        print("8 levels")
        print(gray8)
        print("64 levels")
        print(gray64)
        print("128 levels")
        print(gray128)
        print("256 levels")
        print(gray256)
        
        print("central 64")
        print_middle_values(gray64, 25)
        print("central 128")
        print_middle_values(gray128, 25)
        print("central 256")
        print_middle_values(gray256, 25)

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
        # cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray8.jpg'), gray8)
        # cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray64.jpg'), gray64)
        # cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray128.jpg'), gray128)
        # cv2.imwrite(os.path.join(output_dir, f'{pic_name}_gray256.jpg'), gray256)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
