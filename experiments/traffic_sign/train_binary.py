"""Training script for binary traffic sign classification using Bi-FRN."""

import argparse
import json
import os
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from datasets import dataloaders
from models.FRN import FRN
from utils import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary Bi-FRN classifier")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing 'train' and 'val' folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/traffic_sign",
        help="Directory where checkpoints and logs will be stored.",
    )
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for training.")
    parser.add_argument(
        "--eval-batch-size", type=int, default=64, help="Mini-batch size for evaluation."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="Initial learning rate for Adam optimizer."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay used in the optimizer."
    )
    parser.add_argument(
        "--lr-step", type=int, default=20, help="Epoch interval for learning rate decay."
    )
    parser.add_argument(
        "--lr-gamma", type=float, default=0.2, help="Multiplicative factor for learning rate decay."
    )
    parser.add_argument(
        "--train-transform-type",
        type=int,
        default=0,
        help="Transformation type during training (see datasets.transform_manager).",
    )
    parser.add_argument(
        "--test-transform-type",
        type=int,
        default=0,
        help="Transformation type during evaluation (see datasets.transform_manager).",
    )
    parser.add_argument(
        "--resnet",
        action="store_true",
        help="Use ResNet-12 backbone instead of Conv-4.",
    )
    parser.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable weighted sampling when training on imbalanced data.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for training.",
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default=None,
        help="Optional path to store training and validation metrics as JSON.",
    )
    return parser.parse_args()


def evaluate(model: FRN, loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, label in loader:
            images = images.to(device)
            label = label.to(device)
            log_probs = model.forward_pretrain(images)
            pred = log_probs.argmax(dim=1)
            preds.append(pred.cpu())
            targets.append(label.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    num_classes = loader.dataset.num_classes if hasattr(loader.dataset, "num_classes") else len(loader.dataset.classes)
    cm = metrics.confusion_matrix(preds, targets, num_classes=num_classes)
    precision, recall, f1 = metrics.precision_recall_f1(cm)
    total = cm.sum().item()
    accuracy = cm.diag().sum().float() / total if total > 0 else torch.tensor(0.0)
    macro_f1 = f1.mean() if f1.numel() > 0 else torch.tensor(0.0)

    per_class = {}
    for idx, class_name in enumerate(loader.dataset.classes):
        per_class[class_name] = {
            "precision": float(precision[idx].item()),
            "recall": float(recall[idx].item()),
            "f1": float(f1[idx].item()),
        }

    return {
        "accuracy": float(accuracy.item()),
        "macro_f1": float(macro_f1.item()),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }


def train():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = os.path.join(args.data_root, "train")
    val_dir = os.path.join(args.data_root, "val")

    balanced = not args.no_balanced_sampler
    train_loader, class_weights = dataloaders.classification_train_dataloader(
        data_path=train_dir,
        batch_size=args.batch_size,
        transform_type=args.train_transform_type,
        balanced=balanced,
    )
    val_loader = dataloaders.classification_eval_dataloader(
        data_path=val_dir,
        batch_size=args.eval_batch_size,
        transform_type=args.test_transform_type,
    )

    class_names = train_loader.dataset.classes
    num_classes = len(class_names)

    model = FRN(resnet=args.resnet, is_pretraining=True, num_cat=num_classes)
    model.to(device)

    criterion = nn.NLLLoss(weight=class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    history = {"train": [], "val": []}
    best_macro_f1 = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in epoch_iter:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            log_probs = model.forward_pretrain(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

            preds = log_probs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += images.size(0)

            epoch_iter.set_postfix(
                loss=f"{loss.item():.4f}", acc=f"{(preds == labels).float().mean().item():.4f}"
            )

        scheduler.step()

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)

        val_metrics = evaluate(model, val_loader, device)

        history["train"].append({"loss": train_loss, "acc": train_acc})
        history["val"].append(val_metrics)

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
            f"Val Acc={val_metrics['accuracy']:.4f}, Val Macro-F1={val_metrics['macro_f1']:.4f}"
        )
        for class_name, stats in val_metrics["per_class"].items():
            print(
                f"  {class_name}: precision={stats['precision']:.4f}, "
                f"recall={stats['recall']:.4f}, f1={stats['f1']:.4f}"
            )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                    "macro_f1": best_macro_f1,
                },
                best_path,
            )
            print(f"Saved new best model to {best_path}")

    final_path = os.path.join(args.output_dir, "last_model.pth")
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "args": vars(args),
            "macro_f1": best_macro_f1,
        },
        final_path,
    )
    print(f"Saved final model to {final_path}")

    if args.save_metrics:
        with open(args.save_metrics, "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        print(f"Training history stored at {args.save_metrics}")


if __name__ == "__main__":
    train()

