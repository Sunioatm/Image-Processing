import numpy as np

def sobel_filter(img):
    mask_n = 3
    mask_Gx = np.array([[-1,  0, 1],
                        [-2,  0, 2],
                        [-1,  0, 1]])
    
    mask_Gy = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    
    filtered_img_Gx = np.zeros_like(img, dtype=np.float64)
    filtered_img_Gy = np.zeros_like(img, dtype=np.float64)
    
    m, n = img.shape
    offset = mask_n // 2
    for i in range(offset, m - offset): 
        for j in range(offset, n - offset): 
            mask_area = img[i - offset:i + offset + 1, j - offset:j + offset + 1]
            filtered_img_Gx[i, j] = np.sum(mask_area * mask_Gx)
            filtered_img_Gy[i, j] = np.sum(mask_area * mask_Gy)

    # # Show the filtered images using OpenCV
    # cv2.imshow('Filtered Gx', filtered_img_Gx.astype(np.uint8))
    # cv2.imshow('Filtered Gy', filtered_img_Gy.astype(np.uint8))
    
    # # Wait for a key press and then close all OpenCV windows
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return filtered_img_Gx, filtered_img_Gy

def enhance_with_sobel(img):
    filtered_Gx, filtered_Gy = sobel_filter(img)
    
    gradient_magnitude = np.sqrt(filtered_Gx**2 + filtered_Gy**2)
    
    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude) * 255
    
    sharpened_img = img + gradient_magnitude
    sharpened_img = np.clip(sharpened_img, 0, 255) 
    
    return sharpened_img.astype(np.uint8)
