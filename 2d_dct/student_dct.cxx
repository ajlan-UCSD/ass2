#include "main.h"
#include <iostream>
#include <opencv2/core.hpp> // Include necessary OpenCV headers

using namespace cv;

// Lookup Tables (LUTs) for DCT
Mat LUT_w;
Mat LUT_h;

// Helper function for scaling factor
float sf(int in) {
    if (in == 0)
        return 0.70710678118; // = 1 / sqrt(2)
    return 1.;
}

// Initialize LUT
void initDCT(int WIDTH, int HEIGHT) {
    // Initialize LUT matrices (or any other necessary setup)
    LUT_w = Mat::zeros(WIDTH, WIDTH, CV_32FC1);
    LUT_h = Mat::zeros(HEIGHT, HEIGHT, CV_32FC1);

    // Fill LUT matrices with appropriate cosine values
    for (int i = 0; i < WIDTH; ++i) {
        for (int j = 0; j < WIDTH; ++j) {
            LUT_w.at<float>(i, j) = cos(M_PI / (float)WIDTH * (j + 0.5) * i) * sf(i);
        }
    }
    for (int i = 0; i < HEIGHT; ++i) {
        for (int j = 0; j < HEIGHT; ++j) {
            LUT_h.at<float>(i, j) = cos(M_PI / (float)HEIGHT * (j + 0.5) * i) * sf(i);
        }
    }
}

// Naive DCT Implementation: O(N^4)
Mat student_dct_naive(Mat input) {
    const int HEIGHT = input.rows;
    const int WIDTH  = input.cols;  

    float scale = 2.0 / sqrt(HEIGHT * WIDTH);
    Mat result = Mat(HEIGHT, WIDTH, CV_32FC1);

    float* result_ptr = result.ptr<float>();
    float* input_ptr  = input.ptr<float>();

    for (int x = 0; x < HEIGHT; x++) {
        for (int y = 0; y < WIDTH; y++) {
            float value = 0.f;
            for (int i = 0; i < HEIGHT; i++) {
                for (int j = 0; j < WIDTH; j++) {
                    value += input_ptr[i * WIDTH + j]
                            * cos(M_PI / ((float)HEIGHT) * (i + 1. / 2.) * (float)x)
                            * cos(M_PI / ((float)WIDTH) * (j + 1. / 2.) * (float)y);
                }
            }
            value = scale * sf(x) * sf(y) * value;
            result_ptr[x * WIDTH + y] = value;
        }
    }

    return result;
}

// Block DCT Implementation: Matrix Multiplication
Mat student_dct_block(Mat input) {
    assert(input.rows == input.cols);    
    int N = input.rows;

    std::cout << "Input matrix size: " << input.size() << std::endl;
    std::cout << "LUT_w matrix size: " << LUT_w.size() << std::endl;

    Mat output = Mat::zeros(input.size(), input.type());
    gemm(LUT_w, input, 1, Mat(), 0, output);  // output = LUT_w * input
    gemm(output, LUT_w.t(), 1, Mat(), 0, output);  // output = output * LUT_w.t()

    return output;
}

// Unrolled Block DCT Implementation
Mat student_dct_unrolled(Mat input) {
    assert(input.rows == input.cols);
    int N = input.rows;

    Mat temp = Mat::zeros(input.size(), input.type());
    Mat output = Mat::zeros(input.size(), input.type());

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;            
            for (int k = 0; k < N; k += 4) {
                sum += LUT_w.at<float>(i, k) * input.at<float>(k, j)
                     + LUT_w.at<float>(i, k+1) * input.at<float>(k+1, j)
                     + LUT_w.at<float>(i, k+2) * input.at<float>(k+2, j)
                     + LUT_w.at<float>(i, k+3) * input.at<float>(k+3, j);
            }
            temp.at<float>(i, j) = sum;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k += 4) {
                sum += temp.at<float>(i, k) * LUT_w.at<float>(j, k)
                     + temp.at<float>(i, k+1) * LUT_w.at<float>(j, k+1)
                     + temp.at<float>(i, k+2) * LUT_w.at<float>(j, k+2)
                     + temp.at<float>(i, k+3) * LUT_w.at<float>(j, k+3);
            }
            output.at<float>(i, j) = sum;
        }
    }

    return output;
}

// Placeholder for Neon Block DCT Implementation
Mat student_dct_neon(Mat input) {
    // Implement ARM Neon optimized DCT here
    // Note: This is hardware-dependent and might require cross-compilation
    return student_dct_unrolled(input); // Fallback to unrolled if not using NEON
}

// Wrapper Function for DCT Implementations
cv::Mat student_dct(cv::Mat input, int version) {
    switch (version) {
        case 1:
            return student_dct_block(input);
        case 2:
            return student_dct_unrolled(input);
        case 3:
            return student_dct_neon(input);
        default:
            return student_dct_naive(input);
    }
}

