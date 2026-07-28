from model import ModifiedMobileNetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import optim
import argparse
import csv
from tqdm import tqdm
import torch
from torchvision.transforms import v2 as T
import torch.nn as nn
import os


def build_dataloaders(train_dir, val_dir, batch_size, grayscale, augmentation, num_workers):
    if augmentation:
        train_transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomRotation(degrees=10),
            # ImageFolder will treat images as RGB images, even if they're stored as grayscale.
            T.Grayscale() if grayscale else T.Identity(),
            T.ToImage(),

            # Rescales [0,255] uint8 -> [0,1] float32. Deliberately no Normalize()
            # (no ImageNet mean/std) — keeps the model's expected input as plain
            # [0,1], which is what makes the int8 quantization math simple.
            #
            # Without Normalize() (this pipeline):
            #   int8_value = round(real_value/scale) + zero_point
            #              = round((camera_byte/255) / (1/255)) - 128
            #              = camera_byte - 128
            #   -> one integer subtraction, no floats, same for every channel.
            #
            # With Normalize() (hypothetical, NOT used here):
            #   real_value = (camera_byte/255 - mean[c]) / std[c]
            #   int8_value = round(real_value / scale) + zero_point
            #   -> doesn't simplify; needs float divide, per-channel mean/std
            #      subtraction, then the quantize step — three ops instead of one,
            #      per pixel per channel, every inference.
            T.ToDtype(torch.float32, scale=True),
        ])

        val_transform = T.Compose([
            T.Grayscale() if grayscale else T.Identity(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

    else:
        train_transform = val_transform = T.Compose([
            T.Grayscale() if grayscale else T.Identity(),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    print(f"{train_dataset.classes}, {train_dataset.class_to_idx}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def build_model(alpha, in_channels, num_512_blocks):
    model = ModifiedMobileNetV1(
        alpha=alpha, in_channels=in_channels, num_512_blocks=num_512_blocks)
    return model


def train(model, train_loader, val_loader, device, num_epochs, learning_rate, l2_value, threshold, checkpoint_dir):
    model = model.to(device)

    # Class balance is mild here — no pos_weight or resampling used.
    # Recall bias for the sentry use case is handled at threshold-selection
    # time instead (see evaluate.py), which is free to re-tune without
    # retraining. Revisit pos_weight only if real-hardware fine-tuning
    # data turns out to be heavily imbalanced.
    loss_fn = nn.BCEWithLogitsLoss()

    # Values are adapted from Wake Vision paper.
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=l2_value)

    # LR follows a cosine curve, monotonically decreasing from --lr at
    # epoch 1 down toward 0 at epoch num_epochs — never increases. The
    # *rate* of decrease varies (slow at the start, fastest around the
    # middle, slow again near the end), but the value itself only goes
    # down. The whole curve is shaped around num_epochs — running
    # fewer/more epochs than planned (or stopping early) does NOT give
    # the same schedule as training for that number from the start; the
    # LR trajectory itself is different.
    #
    # No early stopping used: cutting training short would also cut the
    # schedule short, sacrificing the low-LR fine-tuning phase the curve
    # is built around. Instead, best-checkpoint saving (see save_checkpoint)
    # lets the full schedule run and just keeps whichever epoch's weights
    # were actually best — no need to guess the right stopping point.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs)

    train_losses = []
    val_losses = []
    val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], []
    least_val_loss = float("inf")
    os.makedirs(f"{checkpoint_dir}", exist_ok=True)

    csv_path = f"{checkpoint_dir}/metrics.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                "val_loss", "val_acc", "val_precision", "val_recall", "val_f1"])

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        tp = fp = tn = fn = 0

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for imgs, y_actual in progress_bar:
            imgs, y_actual = imgs.to(device), y_actual.to(device)
            y_pred = model(imgs)
            loss = loss_fn(y_pred.squeeze(1), y_actual.float())
            probs = torch.sigmoid(y_pred.squeeze(1))
            preds = (probs >= threshold).long()
            y_actual = y_actual.long()

            tp += ((preds == 1) & (y_actual == 1)).sum().item()
            fp += ((preds == 1) & (y_actual == 0)).sum().item()
            tn += ((preds == 0) & (y_actual == 0)).sum().item()
            fn += ((preds == 0) & (y_actual == 1)).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # TODO: implement balanced accuracy, precision, recall, f1
        train_acc = (tp+tn) / (tp+fp+tn+fn)
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        tqdm.write(
            f"Epoch {epoch+1}\nTrain -> Loss: {avg_train_loss:.4f}, TrainAcc: {train_acc:.4f}")

        avg_val_loss, accuracy, precision, recall, f1 = validate(
            model, val_loader, loss_fn, device, threshold)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch+1, avg_train_loss, train_acc, avg_val_loss, accuracy, precision, recall, f1])

        least_val_loss = save_checkpoint(
            model, avg_val_loss, least_val_loss, checkpoint_dir)

    return model, train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s


def validate(model, val_loader, loss_fn, device, threshold, label="Validation"):
    model.eval()
    val_loss = 0.0
    tp = fp = tn = fn = 0

    progress_bar = tqdm(val_loader, desc="Validation", leave=False)
    with torch.no_grad():
        for imgs, y_actual in progress_bar:
            imgs, y_actual = imgs.to(device), y_actual.to(device)
            y_pred = model(imgs)
            loss = loss_fn(y_pred.squeeze(1), y_actual.float())
            probs = torch.sigmoid(y_pred.squeeze(1))
            preds = (probs >= threshold).long()
            y_actual = y_actual.long()

            tp += ((preds == 1) & (y_actual == 1)).sum().item()
            fp += ((preds == 1) & (y_actual == 0)).sum().item()
            tn += ((preds == 0) & (y_actual == 0)).sum().item()
            fn += ((preds == 0) & (y_actual == 1)).sum().item()
            val_loss += loss.item()

        # TODO: implement balanced accuracy
        accuracy = (tp+tn) / (tp+fp+tn+fn)
        precision = tp / (tp+fp) if (tp+fp) else 0.0
        recall = tp / (tp+fn) if (tp+fn) else 0.0
        f1 = 2*precision*recall / \
            (precision+recall) if (precision+recall) else 0.0
        avg_val_loss = val_loss / len(val_loader)
        tqdm.write(
            f"{label} -> Loss: {avg_val_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return avg_val_loss, accuracy, precision, recall, f1


def save_checkpoint(model, avg_val_loss, least_val_loss, checkpoint_dir):
    if (avg_val_loss < least_val_loss):
        least_val_loss = avg_val_loss
        torch.save(model.state_dict(),
                   f"{checkpoint_dir}/least_val_loss_model.pth")
        tqdm.write(f"Best model saved.\n------------------")
    else:
        tqdm.write("------------------")
    return least_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-dir", default="data/train_quality")
    parser.add_argument("--val-dir", default="data/validation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=4e-6)
    parser.add_argument("--batch-size", type=int, default=128)
    # Only affects which precision/recall/F1 get printed/logged during
    # training — does not affect the loss or optimizer. Real threshold
    # selection happens post-training; see evaluate.py.
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--num-512-blocks", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--checkpoint-dir", default="models")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, val_loader = build_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        grayscale=args.grayscale,
        augmentation=args.augmentation,
        num_workers=args.num_workers)

    in_channels = 1 if args.grayscale else 3

    # TODO: try a different weight init. PyTorch already uses Kaiming
    # uniform by default (fine for ReLU6), but MobileNet's reference
    # implementation uses Kaiming normal with fan_out instead. Small
    # effect expected (maybe faster early convergence) — not a fix
    # for the epoch-12 overfitting, just worth testing separately.
    model = build_model(
        alpha=args.alpha,
        in_channels=in_channels,
        num_512_blocks=args.num_512_blocks)

    _, train_losses, val_losses, val_accuracies, val_precisions, val_recalls, val_f1s = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        l2_value=args.weight_decay,
        threshold=args.threshold,
        checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
