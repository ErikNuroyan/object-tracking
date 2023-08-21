#include "utils.hpp"
#include "video_processor.hpp"

#include <opencv2/highgui.hpp>

VideoProcessor::VideoProcessor(std::string_view filePath, bool demo)
                               : _videoCapture(std::string(filePath)),
                                 _videoWriter(),
                                 _personDetector(),
                                 _poseEstimator(),
                                 _outputBBoxStream(OUTPUTS_PATH "/bboxes.csv"),
                                 _outputLandmarksStream(OUTPUTS_PATH "/landmarks.csv"),
                                 _demo(demo) {
    if (!_videoCapture.isOpened()) {
        std::cerr << "Failed to open video capture" << std::endl;
        exit(1);
    }

    _videoWriter.open(OUTPUTS_PATH "/output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'),_videoCapture.get(cv::CAP_PROP_FPS),
                      cv::Size(_videoCapture.get(cv::CAP_PROP_FRAME_WIDTH), _videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT)));
    if (!_videoWriter.isOpened()) {
        std::cerr << "Failed to open video writer" << std::endl;
        exit(1);
    }

    if (!_outputBBoxStream.is_open()) {
        std::cerr << "Failed to open output bbox stream" << std::endl;
        exit(1);
    }

    if (!_outputLandmarksStream.is_open()) {
        std::cerr << "Failed to open output landmarks stream" << std::endl;
        exit(1);
    }
    initWriters();

    if (_demo) {
        cv::namedWindow("Output");
    }
}

VideoProcessor::~VideoProcessor() {
    _videoCapture.release();
    _videoWriter.release();
    _outputBBoxStream.close();
    _outputLandmarksStream.close();
    cv::destroyWindow("Output");
}

void VideoProcessor::process() {
    cv::Mat frame;
    int frameId = 0;
    while (_videoCapture.read(frame)) {
        PersonDetector::ModelOutput detections = _personDetector.inference(frame);
        cv::Mat finalResult = frame.clone();
        utils::drawBBoxes(finalResult, detections.boundingBoxes, detections.confidenceScores);

        if (!detections.boundingBoxes.empty()) {
            cv::Mat landmarks  = _poseEstimator.inference(frame, detections.boundingBoxes[0]).landmarks;
            cv::Mat roi = finalResult(detections.boundingBoxes[0]);
            utils::drawSkeleton(finalResult, landmarks);
            writeBBoxes(detections.boundingBoxes, detections.confidenceScores, frameId);
            writeLandmarks(landmarks, frameId);
        }

        if (_demo) {
            cv::imshow("Output", finalResult);
            cv::waitKey(1);
        }
        _videoWriter.write(finalResult);
        ++frameId;
    }
}

void VideoProcessor::initWriters() {
    static constexpr std::array<std::string_view, 33> landmarkNames = {
            "nose","left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear","mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder","left_elbow", "right_elbow",
            "left_wrist", "right_wrist","left_pinky", "right_pinky",
            "left_index", "right_index","left_thumb", "right_thumb",
            "left_hip", "right_hip","left_knee", "right_knee",
            "left_ankle", "right_ankle","left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
    };

    for (int i = 0; i < landmarkNames.size(); ++i) {
        const std::string_view& name = landmarkNames[i];
        _outputLandmarksStream << name << "_x," << name << "_y," << name << "_z," << name << "_visibility," << name << "_presence,";
    }
    _outputLandmarksStream << "frame_id" << std::endl;

    _outputBBoxStream << "x,y,width,height,confidence,frame_id" << std::endl;

}

void VideoProcessor::writeBBoxes(const std::vector<cv::Rect>& bboxes, const std::vector<float>& confidenceScores, int frameId) {
    for (int i = 0; i < bboxes.size(); ++i) {
        _outputBBoxStream << bboxes[i].x << ","
                          << bboxes[i].y << ","
                          << bboxes[i].width << ","
                          << bboxes[i].height << ","
                          << confidenceScores[i] << ","
                          << frameId << std::endl;
    }
}

void VideoProcessor::writeLandmarks(const cv::Mat& landmarks, int frameId) {
    constexpr int landmarkStride = 5;
    constexpr int auxLandmarkCount = 6;
    const int finalLandmarkCount = landmarks.cols / landmarkStride - auxLandmarkCount;
    for (int i = 0; i < finalLandmarkCount; ++i) {
        _outputLandmarksStream << landmarks.at<float>(0, i * landmarkStride) << ","
                               << landmarks.at<float>(0, i * landmarkStride + 1) << ","
                               << landmarks.at<float>(0, i * landmarkStride + 2) << ","
                               << utils::sigmoid(landmarks.at<float>(0, i * landmarkStride + 3)) << ","
                               << utils::sigmoid(landmarks.at<float>(0, i * landmarkStride + 4)) << ",";
    }
    _outputLandmarksStream << frameId << std::endl;
}
