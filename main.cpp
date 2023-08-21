#include <iostream>

#include "src/video_processor.hpp"

int main() {
    VideoProcessor processor(VIDEO_PATH, OUTPUTS_PATH);
    processor.process();

    return 0;
}
