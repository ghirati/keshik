# Run this ONCE, after training and hyperparameter tuning are complete.
# Never use test-set results to make decisions — that's what validation is for.

import argparse
import torch
import torch.nn as nn
from train import build_dataloaders, build_model, validate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test-dir", default="data/test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--num-512-blocks", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="models")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    _, test_loader = build_dataloaders(
        train_dir=args.test_dir,   # unused output, but satisfies the signature
        val_dir=args.test_dir,     # this is the one we actually want
        batch_size=args.batch_size,
        grayscale=args.grayscale,
        augmentation=False,        # test set must NEVER be augmented
        num_workers=args.num_workers)

    in_channels = 1 if args.grayscale else 3
    model = build_model(
        alpha=args.alpha,
        in_channels=in_channels,
        num_512_blocks=args.num_512_blocks)
    model = model.to(device)
    model.load_state_dict(torch.load(
        f"{args.checkpoint_dir}/least_val_loss_model.pth", map_location=device))
    loss_fn = nn.BCEWithLogitsLoss()

    avg_test_loss, accuracy, precision, recall, f1 = validate(
        model, test_loader, loss_fn, device, args.threshold, label="Test")


if __name__ == "__main__":
    main()
