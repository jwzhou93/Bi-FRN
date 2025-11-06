"""Evaluate a trained Bi-FRN binary classifier on the validation set."""

import argparse
import json
import os
from typing import Dict

import torch

from datasets import dataloaders
from models.FRN import FRN
from utils import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Bi-FRN binary classifier")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing the validation folder.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint produced by train_binary.py.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Mini-batch size for evaluation."
    )
    parser.add_argument(
        "--transform-type",
        type=int,
        default=0,
        help="Transformation type used for validation images.",
    )
    parser.add_argument(
        "--resnet",
        action="store_true",
        help="Use the ResNet-12 backbone (must match the training configuration).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Optional path to save the evaluation metrics as JSON.",
    )
    return parser.parse_args()


def evaluate(model: FRN, loader, device: torch.device) -> Dict[str, object]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, target in loader:
            images = images.to(device)
            target = target.to(device)
            log_probs = model.forward_pretrain(images)
            pred = log_probs.argmax(dim=1)
            preds.append(pred.cpu())
            targets.append(target.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    num_classes = len(loader.dataset.classes)
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


def main():
    args = parse_args()
    device = torch.device(args.device)

    val_dir = os.path.join(args.data_root, "val")
    loader = dataloaders.classification_eval_dataloader(
        data_path=val_dir,
        batch_size=args.batch_size,
        transform_type=args.transform_type,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = FRN(
        resnet=args.resnet,
        is_pretraining=True,
        num_cat=len(loader.dataset.classes),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    report = evaluate(model, loader, device)

    print("Validation accuracy: {:.4f}".format(report["accuracy"]))
    print("Validation macro F1: {:.4f}".format(report["macro_f1"]))
    print("Confusion matrix:")
    for row in report["confusion_matrix"]:
        print(row)
    for class_name, stats in report["per_class"].items():
        print(
            f"{class_name}: precision={stats['precision']:.4f}, "
            f"recall={stats['recall']:.4f}, f1={stats['f1']:.4f}"
        )

    if args.save_report:
        with open(args.save_report, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2)
        print(f"Saved evaluation report to {args.save_report}")


if __name__ == "__main__":
    main()

