import numpy as np

def averaging_filter(img, mask_n=3):
    mask = np.ones([mask_n,mask_n], dtype=np.float32)
    mask = mask / (mask_n**2)
    filtered_img = np.zeros_like(img)
    m,n = img.shape
    offset = mask_n // 2
    for i in range(offset, m-offset): 
        for j in range(offset, n-offset): 
            mask_area = img[i-offset:i+offset+1, j-offset:j+offset+1]
            filtered_img[i, j]= np.sum(mask_area*mask)

    return filtered_img
