#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <pthread.h>
#include <atomic>

#define RED_WEIGHT      0.2126
#define GREEN_WEIGHT    0.7152
#define BLUE_WEIGHT     0.0722

#define NUM_THREADS 4

typedef struct frames {
    cv::Mat frame;
    cv::Mat gray;
    cv::Mat sobel;
    std::atomic<int> done;
} frame_t;

pthread_barrier_t nextBarrier;
pthread_barrier_t grayBarrier;
pthread_barrier_t sobelBarrier;

frame_t frame_info;
pthread_t threads[NUM_THREADS];
pthread_t mainThread;
int id[NUM_THREADS];

void create_threads();
void *thread_function(void *arg);
void *main_thread(void *arg);
void to442_grayscale(int id, int numRows);
void to442_sobel(int id, int numRows);

void create_threads() {
    for (int i = 0; i < NUM_THREADS; i++) {
        id[i] = i;
        if (pthread_create(&threads[i], NULL, thread_function, (void*)(&id[i])) != 0) {
            perror("pthread_create\n");
            exit(EXIT_FAILURE);
        }
    }
}

void *thread_function(void *arg) {
    int id = *((int*)arg);
    int numRows = frame_info.gray.rows / NUM_THREADS;

    while (true) {
        /* Barrier: wait for new frame */
        pthread_barrier_wait(&nextBarrier);
        if (frame_info.frame.empty() || frame_info.done) break;

        /* Grayscale conversion */
        to442_grayscale(id, numRows);

        /* Wait for all threads to finish grayscale */
        pthread_barrier_wait(&grayBarrier);
        if (frame_info.done) break;

        /* Sobel filter application */
        to442_sobel(id, numRows);

        /* Wait for all threads to finish Sobel */
        pthread_barrier_wait(&sobelBarrier);
        if (frame_info.done) break;
    }
    pthread_exit(NULL);
}

void *main_thread(void *arg) {
    cv::VideoCapture *cap = static_cast<cv::VideoCapture*>(arg);

    if (cap->isOpened()) {
        while (true) {
            cap->read(frame_info.frame);
            if (frame_info.frame.empty()) {
                frame_info.done = 1;
                pthread_barrier_wait(&nextBarrier);
                break;
            }

            // Re-allocate gray and sobel matrices with current frame size
            frame_info.gray.create(frame_info.frame.rows, frame_info.frame.cols, CV_8UC1);
            frame_info.sobel.create(frame_info.frame.rows, frame_info.frame.cols, CV_8UC1);

            pthread_barrier_wait(&nextBarrier);
            pthread_barrier_wait(&grayBarrier);
            pthread_barrier_wait(&sobelBarrier);

            cv::imshow("Sobel", frame_info.sobel);
            if (cv::waitKey(30) == 'q') {
                frame_info.done = 1;
                pthread_barrier_wait(&nextBarrier);
                break;
            }
        }
    }
    pthread_exit(NULL);
}

void to442_grayscale(int id, int numRows) {
    int start = id * numRows;
    int end = (id == NUM_THREADS - 1) ? frame_info.gray.rows : start + numRows;

    for (int y = start; y < end; y++) {
        for (int x = 0; x < frame_info.frame.cols; x++) {
            cv::Vec3b pixel = frame_info.frame.at<cv::Vec3b>(y, x);
            uint8_t grayValue = static_cast<uint8_t>(
                pixel[2] * RED_WEIGHT + pixel[1] * GREEN_WEIGHT + pixel[0] * BLUE_WEIGHT
            );
            frame_info.gray.at<uint8_t>(y, x) = grayValue;
        }
    }
}

void to442_sobel(int id, int numRows) {
    int start = id * numRows;
    int end = (id == NUM_THREADS - 1) ? frame_info.gray.rows : start + numRows;

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    // Ensure bounds are within frame limits
    start = std::max(1, start);
    end = std::min(end, frame_info.gray.rows - 1);

    for (int y = start; y < end; y++) {
        for (int x = 1; x < frame_info.gray.cols - 1; x++) {
            int16_t sumX = 0, sumY = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    uint8_t pixel = frame_info.gray.at<uint8_t>(y + i, x + j);
                    sumX += pixel * Gx[i + 1][j + 1];
                    sumY += pixel * Gy[i + 1][j + 1];
                }
            }

            int magnitude = std::min(255, std::abs(sumX) + std::abs(sumY));
            frame_info.sobel.at<uint8_t>(y, x) = static_cast<uint8_t>(magnitude);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <video_file>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return -1;
    }

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    frame_info.done = 0;

    pthread_barrier_init(&nextBarrier, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&grayBarrier, NULL, NUM_THREADS + 1);
    pthread_barrier_init(&sobelBarrier, NULL, NUM_THREADS + 1);

    create_threads();

    if (pthread_create(&mainThread, NULL, main_thread, &cap) != 0) {
        perror("pthread_create\n");
        return -1;
    }

    pthread_join(mainThread, NULL);
    cap.release();
    cv::destroyAllWindows();

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&nextBarrier);
    pthread_barrier_destroy(&grayBarrier);
    pthread_barrier_destroy(&sobelBarrier);

    return 0;
}

