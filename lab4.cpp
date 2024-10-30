// Lab4.c
// Chris Bae and Cameron Zorio

#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

float R = 0.2126;
float G = 0.7152;
float B = 0.0722;

u_char red;
u_char green;
u_char blue;
u_char gray;

int Gx;
int Gy;

cv::Mat to442_grayscale(cv::Mat &frame);
cv::Mat to442_sobel(cv::Mat &grayFrame);

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

void splitMat(cv::Mat &frame, cv::Mat layers[4]);
cv::Mat stitchMat(cv::Mat layers[4]);

// arguments we'd like to pass each thread
struct ThreadData {
    cv::Mat layer;        // Original layer
    cv::Mat grayFrame;    // Grayscale frame for this layer
    cv::Mat sobelFrame;   // Sobel frame for this layer
};


// declaration for the function each thread will run
// setting the return type as void * will allow the function to return a
// pointer to any type
void* applySobel(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    
    // Convert the current layer to grayscale
    data->grayFrame = to442_grayscale(data->layer);
    
    // Apply the Sobel filter to the grayscale frame
    data->sobelFrame = to442_sobel(data->grayFrame);
    //data->sobelFrame = applyOpenCVSobel(data->grayFrame);
    return nullptr;
}
    
    
// pointers for the return value of each thread
void *thread1Status;
void *thread2Status;

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

    printf("Use 'export DISPLAY=:0' to connect to the Pi's display remotely\n");

    cv::Mat frame, stitchedFrame;
    cv::Mat resultsLayers[4];
    cv::Mat layers[4];
    ThreadData threadData[4]; // Store thread-specific data
    while (true) {
        // Capture frame-by-frame
        videoFile >> frame;
        
    
        // split the current frame into four horizontal layers
        
        splitMat(frame, layers);

        // Create 4 threads, one for each layer
        pthread_t threads[4];
        

        for (int i = 0; i < 4; i++) {
            // give each thread a layer
            threadData[i].layer = layers[i]; 
            threadData[i].grayFrame = cv::Mat(layers[i].rows, layers[i].cols, CV_8UC1);
            threadData[i].sobelFrame = cv::Mat(layers[i].rows - 2, layers[i].cols - 2, CV_8UC1); 
            pthread_create(&threads[i], nullptr, applySobel, &threadData[i]);
        }

        // Wait for all threads to finish
        for (int i = 0; i < 4; i++) {
            pthread_join(threads[i], nullptr);
        }

        // place each processed layer into a single array of Mats
        for(int i = 0; i < 4; i++) {
            resultsLayers[i] = threadData[i].sobelFrame;
        }

        // stitch the layers into a single frame
        stitchedFrame = stitchMat(resultsLayers);

        // Display the frame
        cv::imshow("Frame", stitchedFrame);

        // Press 'q' to break the loop
        if (cv::waitKey(30) == 'q')
            break;
    }

    videoFile.release();
    return 0;
}




cv::Mat to442_grayscale(cv::Mat& frame) {
    /*
    cv::Mat grayFrame;
    cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);
    return grayFrame;
    */

    
    cv::Mat grayFrame = cv::Mat(frame.rows,frame.cols,CV_8UC1);
    int rWeight = static_cast<int>(R * 255);  // scale by 255 for precision
    int gWeight = static_cast<int>(G * 255);
    int bWeight = static_cast<int>(B * 255);
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

                // gray = (float)red * R + (float)green * G + (float)blue * B;
                // grayFrame_i[j] = (u_char)gray;
		grayFrame_i[j] = static_cast<u_char>((red * rWeight + green * gWeight + blue * bWeight) / 255);
        	
            }
        }
    return grayFrame;
    
}

cv::Mat to442_sobel(cv::Mat& grayFrame) {
    cv::Mat sobelFrame = cv::Mat(grayFrame.rows-2,grayFrame.cols-2,CV_8UC1);

    int p11, p12, p13, p21, p23, p31, p32, p33;
    int sum;

    // loop through the grayscale rows, skipping the borders
    for (int i = 1; i < grayFrame.rows-1; i++) 
    {
            // create pointers to the current, previous, and next row
            u_char* grayFrame_im1 = grayFrame.ptr<u_char>(i-1);
            u_char* grayFrame_i = grayFrame.ptr<u_char>(i);
            u_char* grayFrame_ip1 = grayFrame.ptr<u_char>(i+1);

            u_char* sobelFrame_i = sobelFrame.ptr<u_char>(i-1);

            for(int j = 1; j < grayFrame.cols-1; j++)
            {
                // load a local region of pixels and cast to 16 bits
                // LAB 5: explicit, go from double reg to quad reg
                p11 = (int)(grayFrame_im1[j - 1]);
                p12 = (int)(grayFrame_im1[j    ]);
                p13 = (int)(grayFrame_im1[j + 1]); 

                p21 = (int)(grayFrame_i[j - 1]);
                p23 = (int)(grayFrame_i[j + 1]); 

                p31 = (int)(grayFrame_ip1[j - 1]);
                p32 = (int)(grayFrame_ip1[j    ]);
                p33 = (int)(grayFrame_ip1[j + 1]); 

                // apply weights and sum for Gx
                // LAB 5: Load weights into vector reg, vmul, vadd
                Gx = -p11 + p13 - 2*p21 + 2*p23 - p31 + p33;
                Gy = -p11 -2*p12 - p13 + p31 + 2*p32 + p33;
                
            
                // takes abs and add Gx and Gy;
                sum = abs(Gx) + abs(Gy);

                // clamp if necessary
                if (sum > 255) {
                    sum = 255;
                }

                // cast back to 8 bits;
                sum = (u_char)sum;

                // store
                // LAB 5: use vstore
                sobelFrame_i[j-1] = sum;  
            }     
    }
    return sobelFrame;
}

void splitMat(cv::Mat &frame, cv::Mat layers[4]) {
    if (frame.rows % 4 != 0) {
        throw std::runtime_error("Image height is not divisible by 4");
    }
    int layerHeight = frame.rows/4;
    int startRow;
    int stopRow;
    for (int i = 0; i < 4; i++) {
        // find the range of rows to copy over
        startRow = 0 + i*layerHeight;
        stopRow = startRow + layerHeight;

        // "pad" the layers to account for Sobel cropping, padding depends on if it's the 
        // first, last, or middle layers
        
        if(i == 0) {
            stopRow = stopRow + 1;
        } else if (i == 3) {
            startRow = startRow - 1;
        } else {
            startRow = startRow - 1;
            stopRow = stopRow + 1;
        }
        
        layers[i] = frame.rowRange(startRow, stopRow).clone(); 
    }
}

cv::Mat stitchMat(cv::Mat layers[4]) {
    cv::Mat frame = cv::Mat(4*layers[1].rows, layers[1].cols,layers[1].type());
    int layerHeight = layers[1].rows;
    int startRow;
    int stopRow;
    
    for (int i = 0; i < 4; i++) {
        // find the range of rows to copy over
        startRow = 0 + i*layerHeight;
        stopRow = startRow + layerHeight; // rowRange is exclusive 

        // undo the padding
        if(i == 0) {
            stopRow = stopRow - 1;
        } else if (i == 3) {
            startRow = startRow - 1;
            stopRow = startRow + layerHeight - 1;
        } else {
            startRow = startRow - 1;
            stopRow = startRow + layerHeight;
        }
        

        layers[i].copyTo(frame.rowRange(startRow,stopRow));    
    }
    return frame;
}
