#include "fire_detector.hpp"

using namespace std;
using namespace cv;

FireDetector::FireDetector(int threshold, int min_count, double colorVarThresh, int minArea)
    : thresholdDiff(threshold), minCount(minCount),
    colorVarThreshold(colorVarThresh), minArea(minArea), tracker() {
    morphElement = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
}

bool FireDetector::R1(int Cr, int Cb) {
    return (Cr - Cb) > thresholdDiff;
}

bool FireDetector::R2(int Cr, int Cb) {
    return (Cb <= 120) & (Cr >= 150); // Using bitwise AND for faster evaluation
}

bool FireDetector::R3(float V) {
    return V > 0.70f;
}

bool FireDetector::R4(int Y, int Cb, int Cr) {
    return (Cb >= 125) & (Cb <= 135) &
        (Cr >= 125) & (Cr <= 135) &
        (Y >= 230) & (Y <= 240);
}

double computeHistogramVariance(const cv::Mat &channel, const cv::Mat &mask, const cv::Rect &roi) {
    const int histSize = 256; // 8-bit
    std::vector<int> hist(histSize, 0);

    int total = 0;
    double sum = 0.0;

    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        const uchar *ch_row = channel.ptr<uchar>(y);
        const uchar *m_row = mask.ptr<uchar>(y);
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            if (m_row[x]) {
                int val = ch_row[x];
                hist[val]++;
                sum += val;
                total++;
            }
        }
    }

    if (total == 0) return 0.0;

    double mean = sum / total;
    double var = 0.0;

    for (int i = 0; i < histSize; ++i) {
        if (hist[i] > 0) {
            double diff = i - mean;
            var += hist[i] * diff * diff;
        }
    }

    return var / total;
}


vector<vector<Point>> FireDetector::getFireContours(const Mat &hsv, const Mat &ycbcr) {
    vector<Mat> ycbcr_channels;
    split(ycbcr, ycbcr_channels);
    Mat R6_mask(hsv.size(), CV_8UC1, Scalar(0));

    // Pre-calculate constants
    const float v_scale = 1.0f / 255.0f;
    const int rows = hsv.rows;
    const int cols = hsv.cols;

    // Process pixels in parallel (if OpenCV built with TBB)
    parallel_for_(Range(0, rows),
        [&](const Range &range) {
            for (int y = range.start; y < range.end; ++y) {
                const Vec3b *hsv_row = hsv.ptr<Vec3b>(y);
                const Vec3b *ycbcr_row = ycbcr.ptr<Vec3b>(y);
                uchar *mask_row = R6_mask.ptr<uchar>(y);

                for (int x = 0; x < cols; ++x) {
                    const Vec3b &hsv_px = hsv_row[x];
                    const Vec3b &ycbcr_px = ycbcr_row[x];

                    int Y_val = ycbcr_px[0];
                    int Cr = ycbcr_px[1];
                    int Cb = ycbcr_px[2];
                    float V = hsv_px[2] * v_scale;

                    // Standard fire conditions
                    bool standardFire = (R1(Cr, Cb) && R2(Cr, Cb) && R3(V)) || R4(Y_val, Cb, Cr);
                    // Bright flame detection (intensity-driven)
                    bool isBrightFlame = (V > 0.90f);
                    // Hot core detection (for nearly white/yellowish flames)
                    bool isHotCore = (Y_val > 245) && (abs(Cr - Cb) < 10);

                    mask_row[x] = (standardFire || isBrightFlame || isHotCore) ? 255 : 0;
                }
            }
        }
    );

    // Optimized morphology operations
    morphologyEx(R6_mask, R6_mask, MORPH_OPEN, morphElement);
    morphologyEx(R6_mask, R6_mask, MORPH_CLOSE, morphElement);

#ifdef DEBUG
    imshow("Mask (before filter)", R6_mask);
#endif

    // Find and filter contours
    vector<vector<Point>> contours, filteredContours;
    findContours(R6_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat contour_mask(R6_mask.size(), CV_8UC1);
    Mat Cr_roi, Cb_roi;

    for (const auto &contour : contours) {
        Rect rect = boundingRect(contour);
        if (rect.area() < minArea) continue;

        contour_mask.setTo(0);
        drawContours(contour_mask, vector<vector<Point>>{contour}, -1, Scalar(255), FILLED);

        Rect expanded_rect = rect + Size(10, 10);
        expanded_rect &= Rect(0, 0, R6_mask.cols, R6_mask.rows);

        double Cr_variance = computeHistogramVariance(ycbcr_channels[1], contour_mask, expanded_rect);
        double Cb_variance = computeHistogramVariance(ycbcr_channels[2], contour_mask, expanded_rect);

        if (Cr_variance > colorVarThreshold || Cb_variance > colorVarThreshold) {
            filteredContours.push_back(contour);
        }
    }

    return filteredContours;
}

const std::map<int, TrackedObject> &FireDetector::detect(const Mat &frame) {
    Mat flatColor, hsv, ycbcr;

    GaussianBlur(frame, flatColor, Size(5, 5), 0);

    cvtColor(flatColor, hsv, COLOR_BGR2HSV);
    cvtColor(flatColor, ycbcr, COLOR_BGR2YCrCb);

    auto contours = getFireContours(hsv, ycbcr);
    tracker.update(frame, contours);

#ifdef DEBUG
    imshow("Fire Mask", mask);
#endif

    return tracker.getTrackedObjects();
}