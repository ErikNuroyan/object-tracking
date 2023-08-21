#include "person_detector.hpp"

#include <opencv2/imgproc.hpp>

namespace {
    constexpr int InputDim = 320;
    constexpr float InputScale = 1. / 127.5f;
    constexpr float InputMean = 127.5f;
    constexpr float ConfidenceThreshold = 0.65f;
    constexpr float IouThreshold = 0.95f;
}

namespace {
    PersonDetector::ModelOutput prepareModelOutput(const std::vector<cv::Rect>& bboxes,
                                                   const std::vector<int>& indices,
                                                   const std::vector<float>& scores,
                                                   const cv::Size& frameSize) {
        PersonDetector::ModelOutput output;
        for (int i = 0; i < indices.size(); ++i) {
            if (indices[i] != 1) {
                continue;
            }

            // Scaling the detected rect to make the landmarks model work better
            cv::Point topLeft = bboxes[i].tl();
            cv::Point bottomRight = bboxes[i].br();
            cv::Point center = (topLeft + bottomRight) / 2;

            topLeft = 1.3f * (topLeft - center) + center;
            topLeft.x = std::max(topLeft.x, 0);
            topLeft.y = std::max(topLeft.y, 0);
            bottomRight = 1.3f * (bottomRight - center) + center;
            bottomRight.x = std::min(bottomRight.x, frameSize.width - 1);
            bottomRight.y = std::min(bottomRight.y, frameSize.height - 1);

            output.boundingBoxes.emplace_back(topLeft, bottomRight);
            output.confidenceScores.push_back(scores[i]);
        }

        return output;
    }
}

PersonDetector::PersonDetector() : _personDetector(std::string(DETECTION_MODEL_GRAPH_PATH), std::string(DETECTION_MODEL_CONFIG_PATH)) {
    _personDetector.setInputSize({InputDim, InputDim});
    _personDetector.setInputScale(InputScale);
    _personDetector.setInputMean(InputMean);
    _personDetector.setInputSwapRB(true);
}

PersonDetector::ModelOutput PersonDetector::inference(const cv::Mat& frame) {
    ModelOutput modelOutput;
    std::vector<int> classIndices;
    std::vector<float> confidenceScores;
    std::vector<cv::Rect> bboxes;
    _personDetector.detect(frame, classIndices, confidenceScores, bboxes,ConfidenceThreshold, IouThreshold);

    return prepareModelOutput(bboxes, classIndices, confidenceScores, {frame.cols, frame.rows});
}
