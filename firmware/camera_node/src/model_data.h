#pragma once

// model_data.cpp is generated, not tracked in git (see .gitignore).
// Regenerate with (run from repo root, or adjust paths accordingly):
//   xxd -i training/export/tf_model/model_full_integer_quant.tflite \
//     > firmware/camera_node/src/model_data.cpp
// (xxd -i converts the .tflite file's raw bytes into a C array
// declaration + length variable.)
// Then rename the generated array/length variables to g_model /
// g_model_len, and add `alignas(16) const` to the array declaration.

extern const unsigned char g_model[];
extern const unsigned int g_model_len;
