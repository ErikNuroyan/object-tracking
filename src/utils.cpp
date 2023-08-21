#include "utils.hpp"

namespace utils {
    float sigmoid(float val) {
        return 1.f / (1.f + exp(-val));
    }

    void drawBBoxes(cv::Mat& image, const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidenceScores) {
        for (int i = 0; i < bboxes.size(); ++i) {
            const cv::Rect& bbox = bboxes[i];
            cv::rectangle(image, bbox, {0, 0, 255}, 5, cv::LineTypes::LINE_8);
            const cv::Point textShift = (bbox.br() - bbox.tl()) * 0.05f;
            cv::putText(image, "Confidence: " + std::to_string(confidenceScores[i]), bbox.tl() + textShift, cv::FONT_HERSHEY_SIMPLEX, 1.2f, {0, 0, 0}, 3);
        }
    }

    void drawSkeleton(cv::Mat& image, const cv::Mat& landmarks) {
        static constexpr std::array<std::pair<int, int>, 35> edges = {{
           { 0,  1}, { 1,  2}, { 2,  3}, { 3,  7}, { 0,  4}, { 4,  5}, { 5,  6},
           { 6,  8}, { 9, 10}, {11, 13}, {13, 15}, {15, 17}, {15, 21}, {15, 19},
           {17, 19}, {12, 14}, {14, 16}, {16, 22}, {16, 20}, {16, 18}, {18, 20},
           {11, 12}, {11, 23}, {23, 25}, {25, 27}, {27, 29}, {29, 31}, {31, 27},
           {12, 24}, {24, 26}, {26, 28}, {28, 30}, {30, 32}, {32, 28}, {24, 23}
        }};

        for (const auto& [pointIndex1, pointIndex2] : edges) {
            // If any of the edge points is invisible or present
            if (sigmoid(landmarks.at<float>(0, pointIndex1 * 5 + 3)) < 0.7f || sigmoid(landmarks.at<float>(0, pointIndex1 * 5 + 4)) < 0.7f ||
                sigmoid(landmarks.at<float>(0, pointIndex2 * 5 + 3)) < 0.7f || sigmoid(landmarks.at<float>(0, pointIndex2 * 5 + 4)) < 0.7f) {
                continue;
            }

            const int x1 = static_cast<int>(landmarks.at<float>(0, pointIndex1 * 5));
            const int y1 = static_cast<int>(landmarks.at<float>(0, pointIndex1 * 5 + 1));
            const int x2 = static_cast<int>(landmarks.at<float>(0, pointIndex2 * 5));
            const int y2 = static_cast<int>(landmarks.at<float>(0, pointIndex2 * 5 + 1));

            cv::circle(image, {x1, y1}, 3, {0, 255, 0}, 3);
            cv::circle(image, {x2, y2}, 3, {0, 255, 0}, 3);

            cv::line(image, {x1, y1}, {x2, y2}, {255, 0, 0}, 3);
        }
    }
}
