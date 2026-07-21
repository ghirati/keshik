from torchvision.transforms import v2 as T
from torchvision import datasets
import torch
import argparse
import numpy as np
import os


def build_representative_dataset(val_dir, grayscale):
    transform = T.Compose([
        # ImageFolder will treat images as RGB images, even if they're stored as grayscale.
        T.Grayscale() if grayscale else T.Identity(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])

    #  557 person images and 507 not_person images
    rep_dataset = datasets.ImageFolder(val_dir, transform=transform)
    return rep_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default="data/validation")
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--npy_export_path",
                        default="export/calibration_data.npy")
    args = parser.parse_args()

    rep_dataset = build_representative_dataset(args.val_dir, args.grayscale)

    imgs = []
    for img, _ in rep_dataset:
        img = img.numpy().transpose(1, 2, 0)    # converts CHW -> HWC
        imgs.append(img)

    calibration_dataset = np.stack(imgs, axis=0)    # (N, H, W, C) = NHWC
    print(
        f"shape: {calibration_dataset.shape}, dtype: {calibration_dataset.dtype}")

    os.makedirs(os.path.dirname(args.npy_export_path), exist_ok=True)
    np.save(args.npy_export_path, calibration_dataset)
    print(f"Saved → {args.npy_export_path}")


if __name__ == "__main__":
    main()
