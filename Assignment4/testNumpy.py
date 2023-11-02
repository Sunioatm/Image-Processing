import numpy as np
import cv2

# Define the image
image = np.array([[0, 1, 0],
                  [0, 1, 0],
                  [0, 1, 0]], dtype=np.float32)
# image = np.array([[0, 0, 1],
#                   [0, 0, 1],
#                   [0, 0, 1]], dtype=np.float32)
# Compute the 2D DFT
dft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero-frequency component to the center
dft_shifted = np.fft.fftshift(dft)

print("DFT:\n", dft[:,:,0], "\n+j", dft[:,:,1])
print("\nDFT Shifted:\n", dft_shifted[:,:,0], "\n+j", dft_shifted[:,:,1])
