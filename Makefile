# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -O3 -lpthread -ffast-math -ftree-vectorize -mcpu=cortex-a53 
# CFLAGS = -Wall -lpthread -O2
# OpenCV include
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`


SRC = lab5.c
TARGET = lab5.out

# Build the application


$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

