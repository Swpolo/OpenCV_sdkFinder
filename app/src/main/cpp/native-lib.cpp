#include <jni.h>
//#include "native-lib.h"
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <android/log.h>

#define mod0(X,Y) (X != Y ? X : 0)

extern "C" {
    using namespace std;
    using namespace cv;

    void createSdkAreaPattern(Mat &m_io);
    void showSdkContour(Mat canvas, vector<Point> rectangle, Scalar color, int thickness = 1);

    JNIEXPORT void JNICALL
    Java_paul_opencv_1sdkfinder_MainActivity_createGuiSdkCpp(
            JNIEnv *env,
            jobject /* this */,
            jlong addrRgba) {

        Mat &pOutput = *(cv::Mat *) addrRgba;

        const uint16_t width = pOutput.rows;   // X
        const uint16_t height = pOutput.cols;  // Y
        const uint16_t min = (width > height ? height : width);
        const uint16_t max = (width < height ? height : width);

        Rect roi((max - min) >> 1, 0, min, min);
        Mat subMat(pOutput, roi);
        createSdkAreaPattern(subMat);
    }

    JNIEXPORT jboolean JNICALL
    Java_paul_opencv_1sdkfinder_MainActivity_findSdkCpp(
            JNIEnv *env,
            jobject /* this */,
            jlong addrMatGray,
            jlong addrMatOut) {

        bool sdkFound = false;

        Mat &m_base = *(Mat *) addrMatGray;
        Mat &m_out = *(Mat *) addrMatOut;

        int width = m_base.rows;   // X
        int height = m_base.cols;  // Y
        int min = (width > height ? height : width);
        int max = (width < height ? height : width);
        Rect roi((max - min) >> 1, 0, min, min);

        Mat m_sub_base(m_base, roi);
        Mat m_sub_out(m_out, roi);

        const vector<Point> subMat{Point(0, 0), Point(0, min), Point(min, min), Point(min, 0)};
        const double coeff_grid     = 0.5 * 0.5;
        const double coeff_block    = coeff_grid * coeff_grid * 0.3 * 0.3 ;
        const double subMatSize     = contourArea(subMat);
        const double gridSizeMin    = subMatSize * coeff_grid;
        const double blockSizeMin   = subMatSize * coeff_block;

        GaussianBlur(m_sub_base, m_sub_base, Size(9, 9), 0);
        adaptiveThreshold(m_sub_base, m_sub_base, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11,
                          5);
        bitwise_not(m_sub_base, m_sub_base);

        vector<vector<Point>> contours;
        vector<vector<Point>> grid;
        vector<vector<Point>> block;
        vector<vector<Point>> defect;
        vector<Point> approx;
        vector<Vec4i> hierarchy;
        vector<Point> biggestRectangle;
        double biggestRectangleSize = 0.0;

        findContours(m_sub_base, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

        // find all rectangle
        for (unsigned int i = 0; i < contours.size(); i++) {
            approxPolyDP(contours.at(i), approx, 4.0, true);

            if (approx.size() != 4) {
                continue;
            }

            if (!isContourConvex(approx)) {
                continue;
            }

            double rectangle_size = contourArea(approx);

            if (rectangle_size > biggestRectangleSize) {
                biggestRectangleSize = rectangle_size;
                biggestRectangle = approx;
            }

            if (rectangle_size > gridSizeMin) {
                grid.push_back(approx);
            } else if (rectangle_size > blockSizeMin) {
                block.push_back(approx);
            } else {
                defect.push_back(approx);
            }
        }

        for (unsigned int i = 0; i < grid.size(); i++) {
            showSdkContour(m_sub_out, grid.at(i), Scalar(0, 255, 0), 5);
            sdkFound = true;
        }
        for (unsigned int i = 0; i < block.size(); i++) {
            showSdkContour(m_sub_out, block.at(i), Scalar(255, 255, 0), 5);
        }
        for (unsigned int i = 0; i < defect.size(); i++) {
            showSdkContour(m_sub_out, defect.at(i), Scalar(255, 0, 0), 5);
        }

        return (jboolean) sdkFound;
    }

    JNIEXPORT void JNICALL
    Java_paul_opencv_1sdkfinder_MainActivity_addWeightedCpp(
            JNIEnv *env,
            jobject /* this */,
            jlong addrM1,
            jlong addrM2,
            jdouble alpha,
            jdouble beta) {

        Mat &m_1 = *(Mat *) addrM1;
        Mat &m_2 = *(Mat *) addrM2;

        addWeighted(m_1, (double) alpha, m_2, (double) beta, 0.0, m_1);
    }

    void createSdkAreaPattern(Mat &m_io) {

        const Scalar patternColor(0, 255, 0);

        const float coeff_square = 0.9;
        const float coeff_gap = 0.63;
        const float coeff_border = 0.1;

        const int size = (int) (m_io.cols * coeff_square);
        const int size_square = (int) (size * (1 - coeff_border));
        const int size_gap = (int) (size * coeff_gap);

        const int top = (m_io.cols - size) >> 1;
        const int top_square = (m_io.cols - size_square) >> 1;
        const int top_gap = (m_io.cols - size_gap) >> 1;


        rectangle(m_io, Rect(top, top, size, size), patternColor, -1);

        const Rect roi_square(top_square, top_square, size_square, size_square);
        const Rect roi_vert(top, top_gap, size, size_gap);
        const Rect roi_hori(top_gap, top, size_gap, size);
        const Rect roi[3] = {roi_square, roi_vert, roi_hori};

        for (int i = 0; i < 3; i++) {
            Mat square(m_io, roi[i]);
            square = Mat::zeros(square.rows, square.cols, CV_8UC4);
        }
    }

    void showSdkContour(Mat canvas, vector<Point> rectangle, Scalar color, int thickness) {
        for (int i = 0; i < 4; i++) {
            line(canvas, rectangle.at(i), rectangle.at(mod0(i + 1, 4)), color, thickness);
        }
    }
}
