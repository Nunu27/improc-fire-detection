#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <map>
#include "fire_detector.hpp"

using namespace cv;
using namespace std;

#define DEBUG

#define VIDEO "2.mp4"
#define INPUT_PATH "videos/" VIDEO
#define OUTPUT_PATH "output/" VIDEO

#define SHOULD_OUTPUT

int main() {
    VideoCapture cap(INPUT_PATH);
    if (!cap.isOpened()) {
        cerr << "Error opening video stream." << endl;
        return -1;
    }

    double source_fps = cap.get(CAP_PROP_FPS);
    int delay = static_cast<int>(1000.0 / source_fps);

    FireDetector fire_detector;
    double fps = 0;
    double totalProcessTime = 0.0;
    int frameCount = 0;
    const double fpsUpdateInterval = 0.2;

    Mat frame, display;
    std::map<int, TrackedObject> fire_objects;

#ifdef SHOULD_OUTPUT
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    Size frame_size(frame_width, frame_height);

    VideoWriter outputVideo(OUTPUT_PATH, VideoWriter::fourcc('X', '2', '6', '4'), source_fps, frame_size, true);
    if (!outputVideo.isOpened()) {
        cerr << "Error: Could not open video writer" << endl;
        return -1;
    }
#endif

    while (true) {
        cap >> frame;
        if (frame.empty()) {
#ifdef SHOULD_OUTPUT
            break;
#endif

            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }

        auto startTime = chrono::high_resolution_clock::now();
        fire_objects = fire_detector.detect(frame);

        auto endTime = chrono::high_resolution_clock::now();
        double processTime = chrono::duration<double>(endTime - startTime).count();
        totalProcessTime += processTime;
        frameCount++;

        if (totalProcessTime > fpsUpdateInterval) {
            fps = frameCount / totalProcessTime;
            frameCount = 0;
            totalProcessTime = 0;
            printf("FPS: %.1f, Process: %.1fms\n", fps, processTime * 1000);
        }

#if defined(DEBUG) || defined(SHOULD_OUTPUT)
        display = frame.clone();

        for (const auto &[id, obj] : fire_objects) {
            if (obj.lifetime < 10) {
                continue;
            }

            cv::drawContours(display, std::vector<std::vector<cv::Point>>{obj.contour}, -1, { 0, 255, 255 }, 2);
            cv::circle(display, obj.centroid, 3, { 0, 0, 255 }, -1);

            cv::putText(display, "ID: " + std::to_string(id),
                obj.centroid + cv::Point2f(10, -10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, { 255, 255, 255 }, 1);
        }


        string fpsText = format("FPS: %.1f", fps);
        string timeText = format("Process: %.1fms", processTime * 1000);
        putText(display, fpsText, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
        putText(display, timeText, Point(10, 55), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 1);
#endif

#ifdef SHOULD_OUTPUT
        outputVideo.write(display);
#endif

#ifdef DEBUG
        imshow("Fire Tracking", display);
#endif

        if (waitKey(delay) == 27) break;
    }

#ifdef SHOULD_OUTPUT
    outputVideo.release();
#endif

    cap.release();
    destroyAllWindows();
    return 0;
}