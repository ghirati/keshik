from ai_edge_litert.interpreter import Interpreter
from torchvision import datasets
from torchvision.transforms import v2 as T
import torch
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="export/tf_model/model_full_integer_quant.tflite")
    parser.add_argument("--data-dir", default="data/validation")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.model_path)
    # reserves memory for every tensor, based on the graph
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Read scale/zero_point from the .tflite file itself, instead of
    # hardcoding them. If the model is ever retrained and re-exported,
    # these values can change — reading them live means this script
    # always uses the correct numbers for whatever model file it's
    # actually pointed at, instead of silently using stale ones.
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    correct = 0
    total = 0
    for img, label in dataset:
        img_np = img.numpy().transpose(1, 2, 0)  # converts CHW -> HWC

        # Going into the model (real pixel -> int8):
        # input_int8 = round(real_pixel / input_scale) + input_zero_point
        img_np = np.round(img_np / input_scale +
                          input_zero_point).astype(np.int8)
        # HWC -> NHWC (adds the batch dim of 1)
        img_np = np.expand_dims(img_np, axis=0)

        # Copies the prepared image data into the model's input tensor slot
        interpreter.set_tensor(input_details["index"], img_np)
        # Runs the model's forward pass
        interpreter.invoke()
        # Reads the results back from the model's output tensor slot
        out = interpreter.get_tensor(output_details["index"])

        # Coming out of the model (int8 -> real logit):
        # real_logit = (output_int8 - output_zero_point) * output_scale
        logit = (out.astype(np.float32) - output_zero_point) * output_scale
        # sigmoid: maps the real-valued logit to a probability in [0, 1]
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob[0][0] >= args.threshold else 0

        correct += (pred == label)
        total += 1
        if (args.verbose):
            print(f"label={label}, prob={prob[0][0]:.4f}, pred={pred}")

    print(f"Quantized accuracy: {correct/total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
