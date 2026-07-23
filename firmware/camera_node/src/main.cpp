#include <Arduino.h>

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
namespace {
// Model starts empty; raw model bytes get loaded in setup().
const tflite::Model* model = nullptr;
// Interpreter is created later in setup(), once the model and resolver exist.
tflite::MicroInterpreter* interpreter = nullptr;
// Register only the ops this model actually uses.
// tensorflow/lite/micro/all_ops_resolver.h would register every op TFLM
// supports, for convenience, but that costs flash space and RAM for
// ops the model never uses.
tflite::MicroMutableOpResolver<6> micro_op_resolver;
// Tensor arena: fixed-size scratch memory for the interpreter.
// Rather than letting the MCU allocate memory freely at runtime,
// this pre-sized buffer is handed to AllocateTensors(), which divides
// it between everything the model needs (inputs, outputs, activations).
constexpr int kTensorArenaSize =
    140 * 1024;  // measured usage: 122,684 bytes; kept headroom above 120*1024.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace
void setup() {
  Serial.begin(115200);
  delay(2000);
  // Load the model from the raw bytes in model_data.cpp.
  // GetModel() returns a pointer to existing data, not a new object — no `new`
  // needed.
  model = tflite::GetModel(g_model);
  // Confirm this model's schema version matches what this TFLM build expects.
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    while (true) {
      Serial.println("Model schema version mismatch!");
    }
  }
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddRelu6();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddPad();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  // MicroInterpreter ties the model, resolver, and arena together:
  // it reads the model's graph, uses the resolver to find each op's
  // kernel, and manages the arena as data flows through the model.
  interpreter = new tflite::MicroInterpreter(model, micro_op_resolver,
                                             tensor_arena, kTensorArenaSize);
  // AllocateTensors() reads the model's graph, computes every tensor's
  // size, and lays them all out inside tensor_arena.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    while (true) {
      Serial.println("AllocateTensors() failed!");
    }
  }
  Serial.println("Model loaded");
  Serial.print("Arena used bytes: ");
  Serial.println(interpreter->arena_used_bytes());
}
void loop() {
  // Intentionally empty for now — inference will go here.
}
