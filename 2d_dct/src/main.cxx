#include "main.h"
#include "timer.h"
#include <iostream>
#include <opencv2/opencv.hpp>

#define FRAME_NUMBER 5 // Set to 0 or -1 to run a while loop

using namespace std;
using namespace cv;

int main(int argc, const char *argv[]) {
    unsigned int c_start;
    unsigned int opencv_c, student_c;

    cout << "WES237B lab 2" << endl;

    VideoCapture cap("input.raw");

    Mat frame, gray, dct_org, dct_student, dct_block, dct_unrolled, dct_neon, diff_img;
    char key = 0;
    float mse;
    int fps_cnt = 0;

    int WIDTH = 64;
    int HEIGHT = 64;

    // 1 argument on command line: WIDTH = HEIGHT = arg
    if (argc >= 2) {
        WIDTH = atoi(argv[1]);
        HEIGHT = WIDTH;
    }
    // 2 arguments on command line: WIDTH = arg1, HEIGHT = arg2
    if (argc >= 3) {
        HEIGHT = atoi(argv[2]);
    }

    initDCT(WIDTH, HEIGHT);

    float avg_perf = 0.f;
    int count = 0;

#if FRAME_NUMBER <= 0
    while (key != 'q')
#else
    for (int f = 0; f < FRAME_NUMBER; f++)
#endif
    {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        resize(gray, gray, Size(WIDTH, HEIGHT));
        gray.convertTo(gray, CV_32FC1);

        // OpenCV DCT
        dct(gray, dct_org);

        // Naive DCT
        LinuxTimer t;
        dct_student = student_dct(gray, 0); // Naive DCT
        t.stop();
        float myTimer = t.getElapsed();

        absdiff(dct_org, dct_student, diff_img);
        diff_img = diff_img.mul(diff_img);
        Scalar se = sum(diff_img);
        mse = se[0] / ((float)HEIGHT * WIDTH);
        printf("RMSE (Naive): %.4f\n", sqrt(mse));

        cout << "Execute time (Naive): " << (double)myTimer / 1000000000.0 << " seconds" << endl;

        // Block DCT
        t.start();
        dct_block = student_dct(gray, 1);
        t.stop();
        myTimer = t.getElapsed();

        absdiff(dct_org, dct_block, diff_img);
        diff_img = diff_img.mul(diff_img);
        se = sum(diff_img);
        mse = se[0] / ((float)HEIGHT * WIDTH);
        printf("RMSE (Block): %.4f\n", sqrt(mse));

        cout << "Execute time (Block): " << (double)myTimer / 1000000000.0 << " seconds" << endl;

        // Unrolled Block DCT
        t.start();
        dct_unrolled = student_dct(gray, 2);
        t.stop();
        myTimer = t.getElapsed();

        absdiff(dct_org, dct_unrolled, diff_img);
        diff_img = diff_img.mul(diff_img);
        se = sum(diff_img);
        mse = se[0] / ((float)HEIGHT * WIDTH);
        printf("RMSE (Unrolled): %.4f\n", sqrt(mse));

        cout << "Execute time (Unrolled): " << (double)myTimer / 1000000000.0 << " seconds" << endl;

        // Neon Block DCT (fallback to unrolled if not using NEON)
        t.start();
        dct_neon = student_dct(gray, 3);
        t.stop();
        myTimer = t.getElapsed();

        absdiff(dct_org, dct_neon, diff_img);
        diff_img = diff_img.mul(diff_img);
        se = sum(diff_img);
        mse = se[0] / ((float)HEIGHT * WIDTH);
        printf("RMSE (NEON): %.4f\n", sqrt(mse));

        cout << "Execute time (NEON): " << (double)myTimer / 1000000000.0 << " seconds" << endl;

        // Save outputs for further analysis
        imwrite("original_frame_" + to_string(f) + ".png", gray);
        imwrite("idct_output_frame_" + to_string(f) + ".png", dct_student);
        imwrite("block_output_frame_" + to_string(f) + ".png", dct_block);
        imwrite("unrolled_output_frame_" + to_string(f) + ".png", dct_unrolled);
        imwrite("neon_output_frame_" + to_string(f) + ".png", dct_neon);

        // If you want to view the images using an external viewer:
        // system("termux-open idct_output_frame_" + to_string(f) + ".png");
    }

    return 0;
}

