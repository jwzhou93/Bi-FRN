"""Run single image inference with a trained Bi-FRN binary classifier."""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image

from datasets import dataloaders
from datasets import transform_manager
from models.FRN import FRN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a traffic sign is normal or abnormal")
    parser.add_argument("--data-root", type=str, required=True, help="Dataset root with the training folder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--image", type=str, required=True, help="Image to classify.")
    parser.add_argument(
        "--transform-type",
        type=int,
        default=0,
        help="Transformation type that matches evaluation/training.",
    )
    parser.add_argument(
        "--resnet",
        action="store_true",
        help="Use the ResNet-12 backbone (must match training).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to perform inference on.",
    )
    return parser.parse_args()


def load_class_names(data_root: str, transform_type: int):
    train_dir = os.path.join(data_root, "train")
    dataset = dataloaders.get_dataset(
        data_path=train_dir,
        is_training=False,
        transform_type=transform_type,
        pre=None,
    )
    return dataset.classes


def main():
    args = parse_args()
    device = torch.device(args.device)

    class_names = load_class_names(args.data_root, args.transform_type)
    num_classes = len(class_names)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = FRN(resnet=args.resnet, is_pretraining=True, num_cat=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    transform = transform_manager.get_transform(
        is_training=False, transform_type=args.transform_type, pre=None
    )

    image = Image.open(args.image).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model.forward_pretrain(input_tensor)
        probs = F.softmax(log_probs, dim=1).squeeze(0)

    pred_idx = int(probs.argmax().item())
    pred_label = class_names[pred_idx]
    confidence = float(probs[pred_idx].item())

    print(f"Prediction: {pred_label} (confidence={confidence:.4f})")
    for idx, class_name in enumerate(class_names):
        print(f"  {class_name}: {probs[idx].item():.4f}")


if __name__ == "__main__":
    main()

