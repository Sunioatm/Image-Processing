import numpy as np

def median_filter(img, mask_n=3):
    mask = np.ones([mask_n,mask_n], dtype=np.float32)
    filtered_img = np.zeros_like(img)
    m,n = img.shape
    offset = mask_n // 2
    for i in range(offset, m-offset): 
        for j in range(offset, n-offset): 
            mask_area = img[i-offset:i+offset+1, j-offset:j+offset+1]
            mask_flatten = (mask_area*mask).flatten()
            filtered_img[i, j]= np.median(mask_flatten)
    return filtered_img
