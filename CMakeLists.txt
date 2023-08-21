cmake_minimum_required(VERSION 3.25)
project(object_tracking)

set(CMAKE_CXX_STANDARD 17)

set(INPUTS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/inputs)
set(OUTPUTS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/outputs)
set(DETECTION_MODEL_GRAPH_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/models/person_detection/frozen_inference_graph.pb)
set(DETECTION_MODEL_CONFIG_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/models/person_detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt)
set(LANDMARKS_MODEL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/resources/models/pose_estimator/pose_landmark.onnx)

find_package(OpenCV REQUIRED)

add_executable(object_tracking src/person_detector.hpp src/person_detector.cpp
                               src/video_processor.hpp src/video_processor.cpp
                               src/pose_estimator.hpp  src/pose_estimator.cpp
                               src/utils.hpp src/utils.cpp main.cpp)

target_include_directories(object_tracking PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_tracking PRIVATE ${OpenCV_LIBS})
target_compile_definitions(object_tracking PRIVATE
        VIDEO_PATH="${INPUTS_PATH}/tennis.mp4" OUTPUTS_PATH="${OUTPUTS_PATH}"
        DETECTION_MODEL_GRAPH_PATH="${DETECTION_MODEL_GRAPH_PATH}" DETECTION_MODEL_CONFIG_PATH="${DETECTION_MODEL_CONFIG_PATH}"
        LANDMARKS_MODEL_PATH="${LANDMARKS_MODEL_PATH}")
