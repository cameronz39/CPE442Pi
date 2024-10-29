// Lab3.c
// Chris Bae and Cameron Zorio

#include <opencv2/opencv.hpp>
#include <stdio.h>

float R = 0.2126;
float G = 0.7152;
float B = 0.0722;

u_char red;
u_char green;
u_char blue;
u_char gray;

int Gx;
int Gy;
cv::Mat applyOpenCVSobel(const cv::Mat& grayFrame) {
    cv::Mat gradX, gradY;
    cv::Mat absGradX, absGradY;

    // Apply Sobel operator in the X and Y directions
    cv::Sobel(grayFrame, gradX, CV_16S, 1, 0, 3);  // X gradient
    cv::Sobel(grayFrame, gradY, CV_16S, 0, 1, 3);  // Y gradient

    // Take the absolute value and convert to 8-bit for display
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);

    // Calculate the gradient magnitude
    cv::Mat sobelFrame;
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, sobelFrame);

    // Crop to remove 1 row and column from each side
    cv::Rect cropRegion(1, 1, sobelFrame.cols - 2, sobelFrame.rows - 2);
    cv::Mat croppedSobelFrame = sobelFrame(cropRegion).clone();

    return croppedSobelFrame;
}
cv::Mat to442_grayscale(cv::Mat& frame);
cv::Mat to442_sobel(cv::Mat& grayFrame);

int main(int argc, char** argv) {
    // Check if a video file path is provided
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }

    // Open the video file from the command line argument
    cv::VideoCapture videoFile(argv[1]); 
    if (!videoFile.isOpened()) {
        std::cerr << "Error: Could not open video file " << argv[1] << std::endl;
        return -1;
    }
    cv::Mat frame, grayFrame, sobelFrame;
    while (true) {
        // Capture frame-by-frame
        videoFile >> frame;

        // apply the Sobel Filter
        grayFrame = to442_grayscale(frame);
        sobelFrame = applyOpenCVSobel(grayFrame);

        // Display the frame
        cv::imshow("Frame", sobelFrame);

        // Press 'q' to break the loop
        if (cv::waitKey(30) == 'q')
            break;
    }

    videoFile.release();
    
    return 0;
}


cv::Mat to442_grayscale(cv::Mat& frame) {
    cv::Mat grayFrame = cv::Mat(frame.rows,frame.cols,CV_8UC1);
        // Loop through each row
        for (int i = 0; i < frame.rows; i++) 
        {
            // Grab a pointer to each row assuming the element type is Vec3b
            cv::Vec3b* frame_i = frame.ptr<cv::Vec3b>(i);
            u_char* grayFrame_i = grayFrame.ptr<u_char>(i);

            // Loop through each col
            for(int j = 0; j < frame.cols; j++)
            {
                red = frame_i[j][0];
                green = frame_i[j][1];
                blue = frame_i[j][2];

                gray = (float)red * R + (float)green * G + (float)blue * B;
                grayFrame_i[j] = (u_char)gray;
            }
        }
    return grayFrame;
}

cv::Mat to442_sobel(cv::Mat& grayFrame) {
    cv::Mat sobelFrame = cv::Mat(grayFrame.rows-2,grayFrame.cols-2,CV_8UC1);

    // loop through the grayscale frame again, skipping the borders
    for (int i = 1; i < grayFrame.rows-1; i++) 
    {
            // create pointers to the current, previous, and next row
            u_char* grayFrame_im1 = grayFrame.ptr<u_char>(i-1);
            u_char* grayFrame_i = grayFrame.ptr<u_char>(i);
            u_char* grayFrame_ip1 = grayFrame.ptr<u_char>(i+1);

            u_char* sobelFrame_i = sobelFrame.ptr<u_char>(i-1);

            for(int j = 1; j < grayFrame.cols-1; j++)
            {
                // apply the sobel filter equations
                Gx = (int)(-grayFrame_im1[j - 1] + grayFrame_im1[j + 1]  
                 - 2 * grayFrame_i[j - 1] + 2 * grayFrame_i[j + 1]
                 - grayFrame_ip1[j - 1] + grayFrame_ip1[j + 1]);  

                Gy =  (int)(-grayFrame_im1[j - 1] - 2 * grayFrame_im1[j] - grayFrame_im1[j + 1] 
                 + grayFrame_ip1[j - 1] + 2 * grayFrame_ip1[j] + grayFrame_ip1[j + 1]);

                sobelFrame_i[j-1] = std::min(255,abs(Gx) + abs(Gy));
            }     
    }
    return sobelFrame;

}
