import numpy as np

# Filter like black square
def notch_reject_filter(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    mask = np.ones((rows, cols, 2), np.uint8)
    r = width // 2
    
    for i in range(-r, r+1):
        mask[crow - d0 + i, ccol - r : ccol + r + 1] = 0
        mask[crow + d0 + i, ccol - r : ccol + r + 1] = 0
    
    return mask

def gaussian_notch_reject(shape, d0=30, width=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2

    y, x = np.ogrid[:rows, :cols]
    d2_from_center = (x - ccol)**2 + (y - crow)**2
    d2_from_d0_above = (x - ccol)**2 + (y - (crow-d0))**2
    d2_from_d0_below = (x - ccol)**2 + (y - (crow+d0))**2

    mask_above = np.exp(-d2_from_d0_above / (2*width**2))
    mask_below = np.exp(-d2_from_d0_below / (2*width**2))
    
    mask_gaussian = 1 - (mask_above + mask_below)

    mask_gaussian = mask_gaussian[:, :, np.newaxis]

    return mask_gaussian

# Example usage
    # notch70 = gaussian_notch_reject(img.shape, 70, 25)
    # notch140 = gaussian_notch_reject(img.shape, 140, 25)

    # notch = notch70 * notch140
    # fshift = dft_shifted * notch
