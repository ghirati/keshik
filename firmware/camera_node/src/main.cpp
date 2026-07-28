#include <Arduino.h>

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "test_images.h"

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
// measured usage with interpreter->arena_used_bytes(): 122,684 bytes;
// kept headroom above 120*1024.
constexpr int kTensorArenaSize = 140 * 1024;
// alignas(16): the ESP32-S3's vector instructions (used by ESP-NN's
// optimized kernels) operate on 16-byte chunks and need data aligned
// to a 16-byte boundary to use the fast path correctly. Without this,
// tensors inside the arena could end up misaligned, silently falling
// back to slow scalar code or failing outright.
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Runs one inference on `image_data` (must match the model's expected input
// size) and prints the raw output, dequantized probability, and inference
// latency. `label` is just for identifying which image this run was.
void run_inference(const int8_t* image_data, const char* label) {
  // handle to input tensor slot 0 (the only one)
  TfLiteTensor* input = interpreter->input(0);
  // copy image into the model's input memory
  memcpy(input->data.int8, image_data, input->bytes);

  unsigned long start = millis();
  // Runs the model's forward pass
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed!");
    while (true) {
    }
  }
  unsigned long elapsed = millis() - start;
  // handle to output tensor slot 0 (the only one)
  TfLiteTensor* output = interpreter->output(0);
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  // Coming out of the model (int8 -> real logit):
  // real_logit = (output_int8 - output_zero_point) * output_scale
  float logit = (output->data.int8[0] - output_zero_point) * output_scale;
  // sigmoid: maps the real-valued logit to a probability in [0, 1]
  float prob = 1.0 / (1.0 + exp(-logit));

  Serial.print(label);
  Serial.print(" -> Raw: ");
  Serial.print(output->data.int8[0]);
  Serial.print(", Prob: ");
  Serial.print(prob, 4);
  Serial.print(", Time (ms): ");
  Serial.println(elapsed);
}

void setup() {
  Serial.begin(115200);
  delay(5000);  // give the serial monitor time to attach before the first print

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

  // These six ops match what Netron shows in the .tflite graph — not
  // the ONNX graph, which differs after onnx2tf conversion (e.g. Pad
  // wasn't in the ONNX graph at all; onnx2tf introduced it for
  // stride-2 convs). Missing an op here fails at AllocateTensors()
  // with a confusing "op not found" error.
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

  // Sanity check: run inference on two known images (one per class) and
  // confirm the results are directionally correct. Verified against the
  // Python-side quantized model (tflite_evaluate.py) — both match within
  // float-vs-int8 rounding tolerance.
  run_inference(test_image_person, "person");
  run_inference(test_image_not_person, "not_person");
}

void loop() {
  // Intentionally empty for now — camera-triggered inference goes here.
}
