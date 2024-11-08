// Lab5.c
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

int count = 0;
cv::Mat to442_grayscale(cv::Mat &frame);
cv::Mat to442_sobel(cv::Mat &grayFrame);

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
	count++;
	printf("Processing frame %d\n", count);
        // Capture frame-by-frame
        videoFile >> frame;

        if(count == 1) {
		printf("Video file row length %d\n", frame.rows);
	}
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
            vst1_u8(&sobelFrame_i[j - 1], result)
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
	
