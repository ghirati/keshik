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

    # Checkpoint was saved on a CUDA device; this script runs locally with
    # no GPU, so map_location="cpu" is required just to load it. Also fine
    # either way: export does a single forward pass to trace the graph, so
    # there'd be no benefit from GPU even if one were available.
    model.load_state_dict(torch.load(
        args.model_path, map_location="cpu"))
    model.eval()

    torch.onnx.export(
        model,
        torch.randn(1, in_channels, 96, 96),
        args.output,
        input_names=["input"],
        output_names=["output"],
        # opset_version=13: chosen for onnx2tf compatibility — this model uses
        # only long-standardized ops (Conv, ReLU6, GlobalAveragePool, etc.), so
        # a newer opset gives no benefit, and 13 is solidly within onnx2tf's
        # well-tested range.
        opset_version=13,
        # dynamo=False: uses the older ONNX exporter instead of the newer
        # default. The newer exporter only builds at opset 18 and failed when
        # asked to convert down to opset 13 (a bug in that conversion step).
        # The older exporter builds directly at opset 13, no conversion needed.
        dynamo=False,
    )

    onnx.checker.check_model(onnx.load("export/model.onnx"))
    print("Export OK — export/model.onnx")


if __name__ == "__main__":
    main()
