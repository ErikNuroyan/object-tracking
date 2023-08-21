#ifndef SRC_VIDEO_PROCESSOR_HPP
#define SRC_VIDEO_PROCESSOR_HPP

#include "person_detector.hpp"
#include "pose_estimator.hpp"

#include <opencv2/videoio.hpp>

#include <fstream>
#include <string_view>

class VideoProcessor {
public:
    explicit VideoProcessor(std::string_view filePath, bool demo = true);
    ~VideoProcessor();
    void process();

private:
    void initWriters();
    void writeBBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidenceScores, int frameId);
    void writeLandmarks(const cv::Mat& landmarks, int frameId);

private:
    cv::VideoCapture _videoCapture;
    cv::VideoWriter _videoWriter;
    PersonDetector _personDetector;
    PoseEstimator _poseEstimator;
    std::ofstream _outputBBoxStream;
    std::ofstream _outputLandmarksStream;
    bool _demo;
};


#endif
