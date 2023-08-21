#include "pose_estimator.hpp"
#include "utils.hpp"

namespace {
    constexpr int LandmarkModelInputDim = 256;
    constexpr float LandmarkModelInputScale = 1. / 255.f;
}

namespace {
    void refineLandmarks(cv::Mat& landmarks, const cv::Mat& heatmap, const cv::Size& frameDims, const cv::Point2f& offset) {
        cv::Mat heatmapReshaped = heatmap.reshape(1, {LandmarkModelInputDim, LandmarkModelInputDim});
        constexpr int landmarksStride = 5;

        constexpr float refineThreshold = 0.7f;
        constexpr int kernelSize = 7;
        constexpr float kernelOffset = (kernelSize - 1.f) / 2.f;

        for (int i = 0; i < landmarks.cols / landmarksStride; ++i) {
            float& x = landmarks.at<float>(0, i * landmarksStride);
            float& y = landmarks.at<float>(0, i * landmarksStride + 1);

            const int rowStart = static_cast<int>(std::max(0.f, y - kernelOffset));
            const int rowEnd = static_cast<int>(std::min(255.f, y + kernelOffset));
            const int colStart = static_cast<int>(std::max(0.f, x - kernelOffset));
            const int colEnd = static_cast<int>(std::min(255.f, x + kernelOffset));

            float confidenceSum = 0.f;
            float maxConfidence = 0.f;
            float weightedColSum = 0.f;
            float weightedRowSum = 0.f;

            for (int r = rowStart; r <= rowEnd; ++r) {
                for (int c = colStart; c <= colEnd; ++c) {
                    float confidence = utils::sigmoid(heatmapReshaped.at<float>(r, c));
                    maxConfidence = std::max(confidence, maxConfidence);
                    confidenceSum += confidence;
                    weightedRowSum += static_cast<float>(r) * confidence;
                    weightedColSum += static_cast<float>(c) * confidence;
                }
            }

            if (maxConfidence > refineThreshold && confidenceSum > 0.f) {
                x = weightedColSum / confidenceSum;
                y = weightedRowSum / confidenceSum;
            }

            x = x * static_cast<float>(frameDims.width) / LandmarkModelInputDim + offset.x;
            y = y * static_cast<float>(frameDims.height) / LandmarkModelInputDim + offset.y;
        }
    }
}

PoseEstimator::PoseEstimator() : _landmarkDetector(cv::dnn::readNetFromONNX(std::string(LANDMARKS_MODEL_PATH))) {}

PoseEstimator::ModelOutput PoseEstimator::inference(const cv::Mat& frame, const cv::Rect& roi) {
    // Get landmarks and Segmentation
    std::vector<std::vector<cv::Mat>> outputs;
    cv::Mat personImage = frame(roi);
    cv::Mat blob = cv::dnn::blobFromImage(personImage, LandmarkModelInputScale, cv::Size(LandmarkModelInputDim, LandmarkModelInputDim), cv::Scalar(), true, false);
    _landmarkDetector.setInput(blob);
    outputs.clear();
    _landmarkDetector.forward(outputs, {"Identity", "Identity_2"});
    refineLandmarks(outputs[0][0], outputs[1][0], {personImage.cols, personImage.rows}, roi.tl());

    ModelOutput output;
    output.landmarks = outputs[0][0];
    output.segmentation = outputs[1][0];

    return output;
}
