#pragma once

// test_images.cpp is generated, not tracked in git (see .gitignore).
// Regenerate with (run from training/, adjust paths as needed):
//   python image_to_c_array.py --image_path <path/to/person/image.png> \
//     --var-name test_image_person \
//     --output_path ../firmware/camera_node/src/test_images.cpp
//   python image_to_c_array.py --image_path <path/to/not_person/image.png> \
//     --var-name test_image_not_person \
//     --output_path ../firmware/camera_node/src/test_images.cpp --append
// (the second call needs --append, or it overwrites the first array)

#include <cstdint>

constexpr int kTestImageSize = 96 * 96 * 3;

extern const int8_t test_image_person[kTestImageSize];
extern const int8_t test_image_not_person[kTestImageSize];
