#include <opencv2/opencv.hpp>

int main() {
    // Load the image
    cv::Mat image = cv::imread("./asset/cartoon.jpg", cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cout << "Could not open or find the image." << std::endl;
        return -1;
    }

    // Create windows to display images
    cv::namedWindow("8 Gray Levels", cv::WINDOW_NORMAL);
    cv::namedWindow("64 Gray Levels", cv::WINDOW_NORMAL);
    cv::namedWindow("128 Gray Levels", cv::WINDOW_NORMAL);
    cv::namedWindow("256 Gray Levels", cv::WINDOW_NORMAL);

    // Convert the image to different gray levels
    cv::Mat gray8, gray64, gray128, gray256;
    cv::normalize(image, gray8, 0, 7, cv::NORM_MINMAX, CV_8U);
    cv::normalize(image, gray64, 0, 63, cv::NORM_MINMAX, CV_8U);
    cv::normalize(image, gray128, 0, 127, cv::NORM_MINMAX, CV_8U);
    cv::normalize(image, gray256, 0, 255, cv::NORM_MINMAX, CV_8U);

    // Display images
    cv::imshow("8 Gray Levels", gray8);
    cv::imshow("64 Gray Levels", gray64);
    cv::imshow("128 Gray Levels", gray128);
    cv::imshow("256 Gray Levels", gray256);

    // Wait for a key press and close windows
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
