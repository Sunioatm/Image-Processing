import cv2
import matplotlib.pyplot as plt

def histogram(inputImage):
    image_path = inputImage
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    # Plotting the histogram
    plt.plot(hist)
    plt.title('Grayscale Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0,256])
    plt.show()

histogram("./Assignment2/assignment2_image1.jpg")