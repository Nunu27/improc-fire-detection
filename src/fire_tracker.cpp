#include "fire_tracker.hpp"

using namespace std;
using namespace cv;

// Optimization constants
namespace {
    const int MIN_CONTOUR_AREA = 25;
    const float DEFAULT_DIST_THRESHOLD = 50.0f;
    const int DEFAULT_MAX_LOST = 5;
}

struct Detection {
    cv::Point2f centroid;
    cv::Rect bbox;
    std::vector<cv::Point> contour;
};

FireTracker::FireTracker(float distThreshold, int maxLost) :
    distanceThreshold(distThreshold > 0 ? distThreshold : DEFAULT_DIST_THRESHOLD),
    maxLostFrames(maxLost > 0 ? maxLost : DEFAULT_MAX_LOST),
    nextID(0) {

    // Validate and clamp parameters
    distanceThreshold = max(10.0f, min(150.0f, distanceThreshold));
    maxLostFrames = max(1, min(10, maxLostFrames));
}

void FireTracker::update(const cv::Mat &frame, const vector<vector<Point>> &contours) {
    if (frame.empty() || contours.empty()) {
        return;
    }

    // Process contours
    vector<Detection> detections;
    detections.reserve(contours.size());

    for (const auto &contour : contours) {
        if (contourArea(contour) < MIN_CONTOUR_AREA) continue;

        Moments m = moments(contour);
        if (m.m00 < 1.0) continue;

        Rect bbox = boundingRect(contour);
        detections.emplace_back(Detection{
            Point2f(float(m.m10 / m.m00), float(m.m01 / m.m00)),
            bbox,
            contour
            });
    }

    // Track management
    map<int, TrackedObject> newTracked;
    vector<bool> matched(detections.size(), false);

    // Match existing tracks
    for (auto &[id, track] : trackedObjects) {
        float minDist = distanceThreshold;
        int matchIdx = -1;

        for (size_t i = 0; i < detections.size(); ++i) {
            if (matched[i]) continue;

            float dist = norm(track.centroid - detections[i].centroid);
            if (dist < minDist) {
                minDist = dist;
                matchIdx = i;
            }
        }

        if (matchIdx != -1) {
            if (track.lifetime < 10) track.lifetime++;

            newTracked[id] = {
                detections[matchIdx].centroid,
                track.lifetime,
                0,
                detections[matchIdx].bbox,
                detections[matchIdx].contour
            };
            matched[matchIdx] = true;
        } else if (track.lost_count < maxLostFrames) {
            track.lost_count++;
            track.lifetime = 0;
            newTracked[id] = track;
        }
    }

    // Add new detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matched[i]) {
            newTracked[nextID++] = {
                detections[i].centroid,
                0, 0,  // Reset counters
                detections[i].bbox,
                detections[i].contour
            };
        }
    }

    trackedObjects.swap(newTracked);
}

const std::map<int, TrackedObject> &FireTracker::getTrackedObjects() const {
    return trackedObjects;
}