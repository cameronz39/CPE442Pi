#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <atomic>
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

// Control variables
std::atomic<bool> stop(false);
std::atomic<int> finishedCount(0);

// Function prototypes
cv::Mat to442_grayscale(cv::Mat &frame);
cv::Mat to442_sobel(cv::Mat &grayFrame);
void splitMat(cv::Mat &frame, cv::Mat layers[NUM_THREADS]);
cv::Mat stitchMat(cv::Mat layers[NUM_THREADS]);

// Persistent thread function
void* applySobelPersistent(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);

    while (!stop) {
        pthread_mutex_lock(&mutex);

        // Wait until new data is available or stop is set
        while (!data->ready && !stop) {
            pthread_cond_wait(&cond, &mutex);
        }

        if (stop) {
            pthread_mutex_unlock(&mutex);
            break;
        }

        // Process the layer
        data->grayFrame = to442_grayscale(data->layer);
        data->sobelFrame = to442_sobel(data->grayFrame);

        // Mark as done and update the finished counter
        data->ready = false;
        finishedCount++;

        pthread_cond_broadcast(&cond); // Wake main thread if waiting
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
        //  std::cout << "Processing frame " << frameCount << std::endl;

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

        // Break the loop after the 200th frame
        if (frameCount == 100) {
            end = std::chrono::high_resolution_clock::now();
        }

    }

    // Calculate elapsed time
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken to process frames: " << elapsed.count() << " seconds." << std::endl;

    // Stop threads and clean up
    pthread_mutex_lock(&mutex);
    stop = true;
    pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }

    videoFile.release();
    return 0;
}

cv::Mat to442_grayscale(cv::Mat& frame) {
    cv::Mat grayFrame = cv::Mat(frame.rows, frame.cols, CV_8UC1);
    int rWeight = static_cast<int>(R * 255);  
    int gWeight = static_cast<int>(G * 255);
    int bWeight = static_cast<int>(B * 255);

    for (int i = 0; i < frame.rows; i++) {
        cv::Vec3b* frame_i = frame.ptr<cv::Vec3b>(i);
        u_char* grayFrame_i = grayFrame.ptr<u_char>(i);

        for (int j = 0; j < frame.cols; j++) {
            red = frame_i[j][0];
            green = frame_i[j][1];
            blue = frame_i[j][2];
            grayFrame_i[j] = static_cast<u_char>((red * rWeight + green * gWeight + blue * bWeight) / 255);
        }
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
