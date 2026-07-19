# plot_curve.py
import csv
import matplotlib.pyplot as plt

epochs, train_loss, val_loss = [], [], []
with open("results/train_quality_run1_metrics.csv") as f:
    for row in csv.DictReader(f):
        epochs.append(int(row["epoch"]))
        train_loss.append(float(row["train_loss"]))
        val_loss.append(float(row["val_loss"]))

best_epoch = epochs[val_loss.index(min(val_loss))]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, train_loss, label="Train loss", linewidth=2)
ax.plot(epochs, val_loss, label="Validation loss", linewidth=2)
ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
           label=f"Best checkpoint (epoch {best_epoch})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs. validation loss — train_quality run")
ax.legend()
fig.tight_layout()
fig.savefig("results/train_quality_run1_loss_curve.png", dpi=150)
print("Saved to results/train_quality_run1_loss_curve.png")