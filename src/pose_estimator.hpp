#ifndef SRC_POSE_ESTIMATOR_HPP
#define SRC_POSE_ESTIMATOR_HPP

#include <opencv2/dnn.hpp>

class PoseEstimator {
public:
    struct ModelOutput {
        cv::Mat landmarks;
        cv::Mat segmentation;
    };

    PoseEstimator();
    ModelOutput inference(const cv::Mat& frame, const cv::Rect& roi);

private:
    cv::dnn::Net _landmarkDetector;
};

#endif
