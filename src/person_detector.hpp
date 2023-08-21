#ifndef SRC_PERSON_DETECTOR_HPP
#define SRC_PERSON_DETECTOR_HPP

#include <opencv2/dnn.hpp>

class PersonDetector {
public:
    struct ModelOutput {
        std::vector<float> confidenceScores;
        std::vector<cv::Rect> boundingBoxes;
    };

    PersonDetector();
    ModelOutput inference(const cv::Mat& frame);

private:
    cv::dnn::DetectionModel _personDetector;
};

#endif
