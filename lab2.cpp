#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("Polar_orbit.ogv");  // Replace with your .ogv file path
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        cv::imshow("Video", frame);
        if (cv::waitKey(30) >= 0) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
