#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

struct TrackedObject {
    cv::Point2f centroid;
    int lifetime;
    int lost_count;
    cv::Rect bbox;
    std::vector<cv::Point> contour;
};

class FireTracker {
public:
    FireTracker(float distThreshold = 50.0f, int maxLost = 10);

    void update(const cv::Mat &frame, const std::vector<std::vector<cv::Point>> &mask);
    const std::map<int, TrackedObject> &getTrackedObjects() const;

private:
    float distanceThreshold;
    int maxLostFrames;
    int nextID;

    std::map<int, TrackedObject> trackedObjects;
};
