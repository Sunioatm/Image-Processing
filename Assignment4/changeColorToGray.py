import cv2
def change_color_to_gray(img_name_withJPG):
    img = cv2.imread(img_name_withJPG, cv2.IMREAD_GRAYSCALE)
    img_name = img_name_withJPG.split(".")[0]
    cv2.imwrite(f"{img_name}_gray.jpg",img)
    
images = ["flower1.jpg", "fruit.jpg"]
for image in images:
    change_color_to_gray(image)