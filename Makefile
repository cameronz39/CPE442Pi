# Compiler
CC = g++

# Compiler flags
CFLAGS = -Wall -O2 -lpthread

# OpenCV include
OPENCV_FLAGS = `pkg-config --cflags --libs opencv4`


SRC = lab5.c
TARGET = lab5.out

# Build the application


$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(OPENCV_FLAGS)

