# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -O3 -lpthread -ffast-math -ftree-vectorize -mcpu=cortex-a53
# OpenCV include
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`


SRC = lab6.c
TARGET = lab6.out

# Build the application


$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

