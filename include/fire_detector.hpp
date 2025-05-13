#pragma once

#include <opencv2/opencv.hpp>
#include "fire_tracker.hpp"

class FireDetector {
public:
    FireDetector(int threshold = 70, int min_count = 5, double colorVarThresh = 100.0, int minArea = 600);

    const std::map<int, TrackedObject> &detect(const cv::Mat &frame);

private:
    int thresholdDiff;
    int minCount;
    FireTracker tracker;
    int frameCount = 0;
    double colorVarThreshold;
    int minArea;
    cv::Mat morphElement;

    std::vector<std::vector<cv::Point>> getFireContours(const cv::Mat &hsv, const cv::Mat &ycbcr);

    bool R1(int Cr, int Cb);
    bool R2(int Cr, int Cb);
    bool R3(float V);
    bool R4(int Y, int Cb, int Cr);
};