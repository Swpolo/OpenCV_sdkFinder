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
    void orderPoints(Point2f src[], Point2f dst[]);

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
            jlong addrMatOut,
            jboolean joutParam) {

        bool sdkFound = false;
        bool outParam = (bool)joutParam;

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

        const int blurSize = 6;
        Mat dilateElement = getStructuringElement(MORPH_RECT, Size((2 * blurSize)+ 1, (2 * blurSize)+ 1), Point(blurSize, blurSize));
        Mat erodeElement = getStructuringElement(MORPH_RECT, Size((2 * blurSize)+ 1, (2 * blurSize)+ 1), Point(blurSize, blurSize));
        dilate(m_sub_base, m_sub_base, dilateElement);
        erode(m_sub_base, m_sub_base, erodeElement);

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

        if(grid.size() != 0) { sdkFound = true; }

        if( sdkFound == true ){
            Point2f src[4];
            Point2f src_o[4];
            for(int i = 0; i < 4; i++) {
                src[i] = Point2f(biggestRectangle.at(i).x, biggestRectangle.at(i).y);
            }
            Point2f dst[4];
            dst[0] = Point2f(roi.x, roi.y);
            dst[1] = Point2f(roi.x + roi.height, roi.y);
            dst[2] = Point2f(roi.x + roi.height, roi.y + roi.width);
            dst[3] = Point2f(roi.x, roi.y + roi.width);

            orderPoints(src, src_o);

            for(int i = 0; i < 4; i++){
                __android_log_print(ANDROID_LOG_DEBUG,"POLO_D", "src_o(%f,%f) : dst(%f,%f)", src_o[i].x, src_o[i].y, dst[i].x, dst[i].y);
            }

            Mat M = getPerspectiveTransform(src_o, dst);

            warpPerspective(m_sub_out, m_sub_out, M, m_sub_out.size());

            circle(m_sub_out, Point(50,50), 25, Scalar(255,0,0), -1);

            return (jboolean) sdkFound;
        }

        if(outParam == true) {
            for (unsigned int i = 0; i < grid.size(); i++) {
                showSdkContour(m_sub_out, grid.at(i), Scalar(0, 255, 0), 5);
            }
            for (unsigned int i = 0; i < block.size(); i++) {
                showSdkContour(m_sub_out, block.at(i), Scalar(255, 255, 0), 5);
            }
            for (unsigned int i = 0; i < defect.size(); i++) {
                showSdkContour(m_sub_out, defect.at(i), Scalar(255, 0, 0), 5);
            }
//            __android_log_print(ANDROID_LOG_DEBUG,"POLO_D", "We printed the result");
        }
        else {
            cvtColor(m_base, m_out, CV_GRAY2BGRA);
//            __android_log_print(ANDROID_LOG_DEBUG,"POLO_D", "We convert color");
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

    void orderPoints(Point2f src[], Point2f dst[]){
        float dump_a = 0.0f;
        int points[4];

        //find point 1 (origin : smallest sum)
        dump_a = src[0].x + src[0].y;
        points[0] = 0;
        for(int i = 1; i < 4; i++){
            float sum = src[i].x + src[i].y;
            if(sum < dump_a){
                sum = dump_a;
                points[0] = i;
            }
        }

        // find point 3 (biggest sum)
        dump_a = src[0].x + src[0].y;
        points[2] = 0;
        for(int i = 1; i < 4; i++){
            float sum = src[i].x + src[i].y;
            if(sum > dump_a){
                sum = dump_a;
                points[2] = i;
            }
        }

        // find point 2 (biggest x not selected)
        dump_a = 0.0f;
        points[1] = 0;
        for(int i = 0; i < 4 ; i++){
            if(i == points[0] || i == points[2]){
                continue;
            }
            if (dump_a < src[i].x) {
                dump_a = src[i].x;
                points[1] = i;
            }
        }

        // find point 4 (last point)
        for(int i = 0; i < 4; i++){
            if(i == points[0] || i == points[1] || i == points[2]) {
                continue;
            }
            points[3] = i;
        }

        for(int i = 0; i < 4; i++){
            dst[i].x = src[points[i]].x;
            dst[i].y = src[points[i]].y;
        }
    }
}
