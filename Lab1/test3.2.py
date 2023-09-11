import cv2
import numpy as np
import matplotlib.pyplot as plt

def power_law_transformation(image, c, gamma):
    # Normalize the image to [0, 1]
    img_normalized = image/255.0
    # Apply power-law transformation
    img_transformed = c * (img_normalized ** gamma)
    # Convert image back to [0, 255]
    img_transformed = np.clip(img_transformed * 255.0, 0, 255).astype(np.uint8)
    return img_transformed

image_path = "./asset/cartoon.jpg"  # Provide the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale mode

c_values = [0.5, 1, 2]
gamma_values = [0.4, 2.5]

fig, axes = plt.subplots(len(c_values), len(gamma_values), figsize=(10,10))

for i, c in enumerate(c_values):
    for j, gamma in enumerate(gamma_values):
        transformed_image = power_law_transformation(image, c, gamma)
        axes[i, j].imshow(transformed_image, cmap='gray')
        axes[i, j].set_title(f"c={c}, gamma={gamma}")
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()
