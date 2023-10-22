import numpy as np

def laplacian_filter(img):
    mask_n = 3
    mask1 = np.array([[ 0,  1,  0],
                      [ 1, -4 , 1],
                      [ 0,  1,  0]])
    
    mask2 = np.array([[ 1,  1,  1],
                      [ 1, -8,  1],
                      [ 1,  1,  1]])
    
    laplacian1 = np.zeros_like(img, dtype=np.int16)
    laplacian2 = np.zeros_like(img, dtype=np.int16)
    
    m, n = img.shape
    offset = mask_n // 2
    for i in range(offset, m-offset): 
        for j in range(offset, n-offset): 
            mask_area = img[i-offset:i+offset+1, j-offset:j+offset+1]
            laplacian1[i, j] = np.sum(mask_area * mask1)
            laplacian2[i, j] = np.sum(mask_area * mask2)
    
    return laplacian1, laplacian2

def enhance_with_laplacian(img):
    laplacian1, laplacian2 = laplacian_filter(img)
    subtracted_img_1 = img.astype(np.int16) - laplacian1
    subtracted_img_2 = img.astype(np.int16) - laplacian2
    clipped_img_1 = np.clip(subtracted_img_1, 0, 255).astype(np.uint8)
    clipped_img_2 = np.clip(subtracted_img_2, 0, 255).astype(np.uint8)
    return clipped_img_1, clipped_img_2
