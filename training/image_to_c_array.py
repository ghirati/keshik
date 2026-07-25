from torchvision.transforms import v2 as T
import torch
import argparse
import numpy as np
import os
from PIL import Image
from ai_edge_litert.interpreter import Interpreter


def build_representative_image(image_path, grayscale):
    img = Image.open(image_path).convert("RGB")

    transform = T.Compose([
        T.Grayscale() if grayscale else T.Identity(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    img_tensor = transform(img)
    return img_tensor


def get_input_quantization(model_path):
    # Read scale/zero_point directly from the .tflite file rather than
    # hardcoding them — guarantees this always matches whatever model
    # is actually being targeted, even if it's re-exported/re-quantized later.
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    return input_details["quantization"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--var-name", required=True)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument(
        "--model-path", default="export/tf_model/model_full_integer_quant.tflite")
    parser.add_argument("--append", action="store_true")
    parser.add_argument(
        "--output_path", default="../firmware/camera_node/src/test_images.cpp")
    args = parser.parse_args()

    img = build_representative_image(args.image_path, args.grayscale)

    # converts CHW -> HWC (96, 96, 3), matching the model's expected input layout
    img_np = img.numpy().transpose(1, 2, 0)

    input_scale, input_zero_point = get_input_quantization(args.model_path)

    # Quantize to int8 using the model's own scale/zero_point.
    # np.round() matters here — plain .astype(int8) truncates toward zero,
    # introducing a small systematic bias into every pixel.
    img_np = np.round(img_np / input_scale + input_zero_point).astype(np.int8)
    flat = img_np.flatten()
    print(
        f"shape: {img_np.shape}, dtype: {img_np.dtype}")

    # Format as a flat C array literal matching model_data.cpp's convention
    values_str = ", ".join(str(v) for v in flat)
    # Build the full declaration
    array_decl = f"alignas(16) const int8_t {args.var_name}[] = {{{values_str}}};\n"

    # First call (no --append) creates the file fresh and writes the include;
    # subsequent calls append additional arrays into the same file without
    # repeating the include. Call once per test image, second call onward
    # with --append.
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.output_path, mode) as f:
        if mode == "w":
            f.write('#include "test_images.h"\n\n')
        f.write(array_decl)
        print(
            f"Wrote {args.var_name}[{flat.shape[0]}] to {args.output_path} (mode={mode})")


if __name__ == "__main__":
    main()
