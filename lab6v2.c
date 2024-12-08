#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <iostream>
#include <chrono>


float R = 0.2126;
float G = 0.7152;
float B = 0.0722;

u_char red;
u_char green;
u_char blue;
u_char gray;

int Gx;
int Gy;

// Thread-related globals
struct ThreadData {
    cv::Mat layer;         // Original layer for this frame
    cv::Mat grayFrame;     // Grayscale of that layer
    cv::Mat sobelFrame;    // Sobel result of that layer
    bool ready;            // Indicates whether this thread has new data to process
};

const int NUM_THREADS = 4;
ThreadData threadData[NUM_THREADS];

// Synchronization variables
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t stopMutex = PTHREAD_MUTEX_INITIALIZER;

// Control variables
bool stop = false;
int finishedCount = 0;

// Function prototypes
cv::Mat to442_grayscale(cv::Mat &frame);
cv::Mat to442_sobel(cv::Mat &grayFrame);
void splitMat(cv::Mat &frame, cv::Mat layers[NUM_THREADS]);
cv::Mat stitchMat(cv::Mat layers[NUM_THREADS]);

// Persistent thread function
void* applySobelPersistent(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    while (true) {
        // Check the stop flag
        pthread_mutex_lock(&stopMutex);
        if (stop) {
            pthread_mutex_unlock(&stopMutex);
            break;
        }
        pthread_mutex_unlock(&stopMutex);

        pthread_mutex_lock(&mutex);
        while (!data->ready) {
            pthread_cond_wait(&cond, &mutex);
            pthread_mutex_lock(&stopMutex);
            if (stop) {
                pthread_mutex_unlock(&stopMutex);
                pthread_mutex_unlock(&mutex);
                return nullptr;
            }
            pthread_mutex_unlock(&stopMutex);
        }

        // Process the layer
        data->grayFrame = to442_grayscale(data->layer);
        data->sobelFrame = to442_sobel(data->grayFrame);
        data->ready = false;

        // Increment finishedCount
        finishedCount++;
        pthread_cond_broadcast(&cond);
        pthread_mutex_unlock(&mutex);
    }

    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file_path>" << std::endl;
        return -1;
    }

    cv::VideoCapture videoFile(argv[1]);
    if (!videoFile.isOpened()) {
        std::cerr << "Error: Could not open video file " << argv[1] << std::endl;
        return -1;
    }

    cv::Mat frame, stitchedFrame;
    cv::Mat layers[NUM_THREADS];
    cv::Mat resultsLayers[NUM_THREADS];

    // Create threads
    pthread_t threads[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i].ready = false;
        pthread_create(&threads[i], nullptr, applySobelPersistent, &threadData[i]);
    }

    int frameCount = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

    while (frameCount <= 100) {
        // Read a frame
        videoFile >> frame;
        if (frame.empty()) break; // End of video

        frameCount++;

        // Split frame into layers
        splitMat(frame, layers);

        // Prepare threads for processing
        pthread_mutex_lock(&mutex);
        finishedCount = 0; // Reset finished count
        for (int i = 0; i < NUM_THREADS; i++) {
            threadData[i].layer = layers[i].clone(); // Assign the layer
            threadData[i].ready = true;             // Mark as ready
        }
        pthread_cond_broadcast(&cond); // Wake up all threads
        pthread_mutex_unlock(&mutex);

        // Wait for all threads to finish
        pthread_mutex_lock(&mutex);
        while (finishedCount < NUM_THREADS) {
            pthread_cond_wait(&cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        // Gather results
        for (int i = 0; i < NUM_THREADS; i++) {
            resultsLayers[i] = threadData[i].sobelFrame.clone();
        }

        // Stitch results and display
        stitchedFrame = stitchMat(resultsLayers);
        cv::imshow("Frame", stitchedFrame);

        // Break on 'q'
        if (cv::waitKey(30) == 'q') break;

        // Start the timer after the first frame
        if (frameCount == 1) {
            start = std::chrono::high_resolution_clock::now();
        }

        // Break the loop after the 100th frame
        if (frameCount == 100) {
            end = std::chrono::high_resolution_clock::now();
        }
    }

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to process frames: " << elapsed.count() << " seconds." << std::endl;

    // Stop threads and clean up
    pthread_mutex_lock(&stopMutex);
    stop = true;
    pthread_mutex_unlock(&stopMutex);
    pthread_cond_broadcast(&cond);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    videoFile.release();
    return 0;
}

cv::Mat to442_grayscale(cv::Mat& frame) {
    cv::Mat grayFrame = cv::Mat(frame.rows, frame.cols, CV_8UC1); // Create output gray frame

    int numCols = frame.cols;
    int num_pixels = frame.rows * numCols; // Total number of pixels
    uint8_t* pixel_pointer = frame.data;  // Pointer to the original frame data
    uint8_t* gray_pointer = grayFrame.data; // Pointer to the grayscale frame data

    // Vectorized weights for RGB to grayscale conversion
    const uint8x16_t r_weight = vdupq_n_u8(54);  // 0.2126 * 256 ≈ 54
    const uint8x16_t g_weight = vdupq_n_u8(183); // 0.7152 * 256 ≈ 183
    const uint8x16_t b_weight = vdupq_n_u8(18);  // 0.0722 * 256 ≈ 18

    for (int i = 0; i < num_pixels; i += 16) {
        // Load 16 pixels (3 channels each) into NEON registers
        uint8x16x3_t rgbVector = vld3q_u8(&pixel_pointer[i * 3]);

        // Multiply each channel by its respective weight
        uint16x8_t redHigh = vmull_u8(vget_high_u8(rgbVector.val[0]), vget_high_u8(r_weight));
        uint16x8_t redLow = vmull_u8(vget_low_u8(rgbVector.val[0]), vget_low_u8(r_weight));

        uint16x8_t greenHigh = vmull_u8(vget_high_u8(rgbVector.val[1]), vget_high_u8(g_weight));
        uint16x8_t greenLow = vmull_u8(vget_low_u8(rgbVector.val[1]), vget_low_u8(g_weight));

        uint16x8_t blueHigh = vmull_u8(vget_high_u8(rgbVector.val[2]), vget_high_u8(b_weight));
        uint16x8_t blueLow = vmull_u8(vget_low_u8(rgbVector.val[2]), vget_low_u8(b_weight));

        // Sum up the weighted values
        uint16x8_t grayHigh = vaddq_u16(vaddq_u16(redHigh, greenHigh), blueHigh);
        uint16x8_t grayLow = vaddq_u16(vaddq_u16(redLow, greenLow), blueLow);

        // Scale down by 256 (right shift by 8) and narrow to 8-bit
        uint8x8_t gray_high_narrow = vshrn_n_u16(grayHigh, 8);
        uint8x8_t gray_low_narrow = vshrn_n_u16(grayLow, 8);

        // Combine the high and low parts into a single vector
        uint8x16_t grayVector = vcombine_u8(gray_low_narrow, gray_high_narrow);

        // Store the result in the grayscale frame
        vst1q_u8(&gray_pointer[i], grayVector);
    }

    return grayFrame;
}


cv::Mat to442_sobel(cv::Mat& grayFrame) {
    cv::Mat sobelFrame = cv::Mat(grayFrame.rows-2,grayFrame.cols-2,CV_8UC1);
    int sum;
    int16x8_t weight_neg1 = vdupq_n_s16(-1);
    int16x8_t weight_pos2 = vdupq_n_s16(2);
    int16x8_t weight_neg2 = vdupq_n_s16(-2);

    // loop through the grayscale rows, skipping the borders
    for (int i = 1; i < grayFrame.rows-1; i++) 
    {
            // create pointers to the current, previous, and next row
            u_char* grayFrame_im1 = grayFrame.ptr<u_char>(i-1);
            u_char* grayFrame_i = grayFrame.ptr<u_char>(i);
            u_char* grayFrame_ip1 = grayFrame.ptr<u_char>(i+1);

            u_char* sobelFrame_i = sobelFrame.ptr<u_char>(i-1);

	   // Loop through the current row in mutiples of 8 now
            for(int j = 1; j < grayFrame.cols-1; j += 8)
            {
	    // Load local region of pixels in vector form
            uint8x8_t p11 = vld1_u8(&grayFrame_im1[j - 1]);
            uint8x8_t p12 = vld1_u8(&grayFrame_im1[j]);
            uint8x8_t p13 = vld1_u8(&grayFrame_im1[j + 1]);

            uint8x8_t p21 = vld1_u8(&grayFrame_i[j - 1]);
            uint8x8_t p23 = vld1_u8(&grayFrame_i[j + 1]);

            uint8x8_t p31 = vld1_u8(&grayFrame_ip1[j - 1]);
            uint8x8_t p32 = vld1_u8(&grayFrame_ip1[j]);
            uint8x8_t p33 = vld1_u8(&grayFrame_ip1[j + 1]);

            // Widen to 16-bit integers for Gx and Gy computation and treat as signed
            int16x8_t p11_16 = vreinterpretq_s16_u16(vmovl_u8(p11));
            int16x8_t p12_16 = vreinterpretq_s16_u16(vmovl_u8(p12));
            int16x8_t p13_16 = vreinterpretq_s16_u16(vmovl_u8(p13));
            int16x8_t p21_16 = vreinterpretq_s16_u16(vmovl_u8(p21));
            int16x8_t p23_16 = vreinterpretq_s16_u16(vmovl_u8(p23));
            int16x8_t p31_16 = vreinterpretq_s16_u16(vmovl_u8(p31));
            int16x8_t p32_16 = vreinterpretq_s16_u16(vmovl_u8(p32));
            int16x8_t p33_16 = vreinterpretq_s16_u16(vmovl_u8(p33));

            // Calculate Gx: -p11 + p13 - 2*p21 + 2*p23 - p31 + p33
            // Apply weights for Gx using element-wise multiplication
            int16x8_t Gx_p11 = vmulq_s16(p11_16, weight_neg1);
            int16x8_t Gx_p21 = vmulq_s16(p21_16, weight_neg2);
            int16x8_t Gx_p23 = vmulq_s16(p23_16, weight_pos2);
            int16x8_t Gx_p31 = vmulq_s16(p31_16, weight_neg1);


            // Sum for Gx
	    int16x8_t Gx = vaddq_s16(Gx_p11, p13_16);
            Gx = vaddq_s16(Gx, Gx_p21);
            Gx = vaddq_s16(Gx, Gx_p23);
            Gx = vaddq_s16(Gx, Gx_p31);
            Gx = vaddq_s16(Gx, p33_16);

            // Calculate Gy: p11 + 2*p12 + p13 - p31 - 2*p32 - p33
            // Apply weights for Gy using element-wise multiplication
	    int16x8_t Gy_p12 = vmulq_s16(p12_16, weight_pos2);
	    int16x8_t Gy_p31 = vmulq_s16(p31_16, weight_neg1);
	    int16x8_t Gy_p32 = vmulq_s16(p32_16, weight_neg2);
	    int16x8_t Gy_p33 = vmulq_s16(p33_16, weight_neg1);

	    // Sum for Gy
	    int16x8_t Gy = vaddq_s16(p11_16, Gy_p12);
	    Gy = vaddq_s16(Gy, p13_16);
	    Gy = vaddq_s16(Gy, Gy_p31);
 	    Gy = vaddq_s16(Gy, Gy_p32);
	    Gy = vaddq_s16(Gy, Gy_p33);

            // Compute the absolute value of Gx and Gy
            int16x8_t abs_Gx = vabsq_s16(Gx);
            int16x8_t abs_Gy = vabsq_s16(Gy);

            // Compute sum = abs(Gx) + abs(Gy)
            int16x8_t sum = vaddq_s16(abs_Gx, abs_Gy);

            // Clamp the values to 255:
            // reinterpret as unsigned, use vmin to clamp to 255, and return to double register
            uint8x8_t result = vqmovn_u16(vminq_u16(vreinterpretq_u16_s16(sum), vdupq_n_u16(255)));

            // Store the result back into the output frame
            vst1_u8(&sobelFrame_i[j - 1], result);
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
