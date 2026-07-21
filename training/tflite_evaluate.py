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
    args = parser.parse_args()

    interpreter = Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    dataset = datasets.ImageFolder(args.data_dir, transform=transform)

    correct = 0
    total = 0
    for img, label in dataset:
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.round(img_np / input_scale +
                          input_zero_point).astype(np.int8)
        img_np = np.expand_dims(img_np, axis=0)

        interpreter.set_tensor(input_details["index"], img_np)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details["index"])

        logit = (out.astype(np.float32) - output_zero_point) * output_scale
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob[0][0] >= args.threshold else 0

        correct += (pred == label)
        total += 1

    print(f"Quantized accuracy: {correct/total:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
