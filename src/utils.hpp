#ifndef SRC_UTILS_HPP
#define SRC_UTILS_HPP

#include <opencv2/opencv.hpp>

namespace utils {
    // Math utils
    float sigmoid(float val);

    // Drawing utils
    void drawBBoxes(cv::Mat& image, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidenceScores);
    void drawSkeleton(cv::Mat& image, const cv::Mat& landmarks);
}

#endif
