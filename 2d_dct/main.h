#ifndef main_h
#define main_h

#include <iostream>
#include <stdio.h>
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

// Function declarations for different DCT implementations
cv::Mat student_dct(cv::Mat input, int version);  // Updated to accept an additional version argument
void initDCT(int WIDTH, int HEIGHT);

// Additional utility functions if needed
void printMatrixSizes(const cv::Mat& input, const cv::Mat& LUT_w);

#endif /* main_h */

