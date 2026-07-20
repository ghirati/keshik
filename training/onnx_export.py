import torch
import argparse
import os
import onnx
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--num-512-blocks", type=int, default=5)
    parser.add_argument(
        "--model-path", default="models/least_val_loss_model.pth")
    parser.add_argument("--output", default="export/model.onnx")
    parser.add_argument("--grayscale", action="store_true")
    args = parser.parse_args()

    os.makedirs("export", exist_ok=True)

    in_channels = 1 if args.grayscale else 3
    model = build_model(
        alpha=args.alpha,
        in_channels=in_channels,
        num_512_blocks=args.num_512_blocks)

    model.load_state_dict(torch.load(
        args.model_path, map_location="cpu"))
    model.eval()

    torch.onnx.export(
        model,
        torch.randn(1, in_channels, 96, 96),
        args.output,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load("export/model.onnx"))
    print("Export OK — export/model.onnx")


if __name__ == "__main__":
    main()
