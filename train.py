"""
train.py — BƯỚC 4 (Huấn luyện mô hình)

Chạy: python train.py --data-root data --model baseline --epochs 30
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import FakeDetectorCNN, FakeDetectorDualBranch


def get_model(name: str):
    if name == "baseline":
        return FakeDetectorCNN()
    elif name == "dual_branch":
        return FakeDetectorDualBranch()
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--model", type=str, default="baseline",
                         choices=["baseline", "dual_branch"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7,
                         help="early stopping patience trên val loss")
    parser.add_argument("--out", type=str, default="best_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, test_loader, class_names = get_dataloaders(
        args.data_root, batch_size=args.batch_size
    )

    model = get_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | {dt:.1f}s")

        # Checkpoint + early stopping dựa trên val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "class_names": class_names,
                "val_acc": val_acc,
                "epoch": epoch,
            }, args.out)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping tại epoch {epoch} (không cải thiện {args.patience} epoch liên tiếp)")
                break

    print(f"\nHoàn tất. Best model lưu tại: {args.out}")
    return history


if __name__ == "__main__":
    main()
