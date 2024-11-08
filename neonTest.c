#include <opencv2/opencv.hpp>
#include <iostream>
#include <arm_neon.h>

int main(int argc, char** argv) {
	// Original 8-element uint8_t array
        uint8_t data[8] = {1, 2, 3, 4, 5, 6, 7, 8};

        // Load 8 uint8 values into a NEON 64-bit register
        uint8x8_t vector8 = vld1_u8(data);

        // Widen to 8 uint16 values in a 128-bit register with zero-extension
        uint16x8_t vector16 = vmovl_u8(vector8);

        // Now `vector16` holds the values as 16-bit integers:
        // (1, 2, 3, 4, 5, 6, 7, 8) with each as a 16-bit integer (uint16_t)

	// create a vector to be added
	uint16x8_t vector2 = vdupq_n_u16(10);

	// add the vectors
	uint16x8_t result_vector = vaddq_u16(vector16,vector2);

	// store the result in memory
	uint16_t result[8];
	vst1q_u16(result,result_vector);

	// Print the results
        printf("Resulting array after addition:\n");
        for (int i = 0; i < 8; i++) {
        	printf("%d ", result[i]);
    	}
	printf("\n");

	return 0;
}
