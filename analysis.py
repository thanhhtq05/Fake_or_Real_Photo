"""
analysis.py — BƯỚC 7 (Phân tích kết quả)

2 công cụ:
1. find_worst_errors: liệt kê những ảnh model sai với confidence cao nhất
   -> giúp phát hiện pattern lỗi (loại fake nào khó, hay model học "shortcut" nào).
2. GradCAM: xem model đang "nhìn" vào đâu trên ảnh khi ra quyết định
   -> kiểm tra model có học đúng feature ảnh, hay đang bám vào artifact
      không liên quan (ví dụ viền ảnh, watermark).

Chạy: python analysis.py --checkpoint best_model.pt --data-root data
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from model import FakeDetectorCNN, FakeDetectorDualBranch
from transforms import MEAN, STD


@torch.no_grad()
def find_worst_errors(model, loader, device, top_k=12):
    """Tìm các mẫu bị dự đoán sai với confidence cao nhất (model 'tự tin nhầm')."""
    model.eval()
    records = []  # (confidence_wrong, image_tensor, true_label, pred_label)

    for images, labels in loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1).cpu()
        confidences = probs.max(dim=1).values.cpu()

        wrong_mask = preds != labels
        for i in torch.where(wrong_mask)[0]:
            records.append((
                confidences[i].item(),
                images[i].clone(),
                labels[i].item(),
                preds[i].item(),
            ))

    records.sort(key=lambda r: r[0], reverse=True)
    return records[:top_k]


def denormalize(img_tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = img_tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_worst_errors(records, class_names, out_path="worst_errors.png"):
    n = len(records)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()

    for i, (conf, img, true_label, pred_label) in enumerate(records):
        axes[i].imshow(denormalize(img))
        axes[i].set_title(
            f"True:{class_names[true_label]} Pred:{class_names[pred_label]}\nconf={conf:.2f}",
            fontsize=8, color="red"
        )
        axes[i].axis("off")

    for j in range(len(records), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Đã lưu {out_path}")


class GradCAM:
    """Grad-CAM cho CNN cuối cùng conv layer trước global pooling."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor.unsqueeze(0))
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # global avg pool gradients
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_baseline.pt")
    parser.add_argument("--data-root", type=str, default="data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]

    model_cls = FakeDetectorCNN if ckpt["model_name"] == "baseline" else FakeDetectorDualBranch
    model = model_cls().to(device)
    model.load_state_dict(ckpt["model_state"])

    _, test_loader, _ = get_dataloaders(args.data_root, handle_imbalance=False)

    # 1. Error analysis
    worst = find_worst_errors(model, test_loader, device, top_k=12)
    plot_worst_errors(worst, class_names)

    # 2. Grad-CAM trên 1 ảnh sai điển hình
    if worst:
        target_layer = model.features[-1].block[-2]  # conv layer cuối trước pool
        cam_tool = GradCAM(model, target_layer)
        conf, img, true_label, pred_label = worst[0]
        cam = cam_tool.generate(img.to(device), pred_label)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(denormalize(img))
        axes[0].set_title("Ảnh gốc")
        axes[0].axis("off")
        axes[1].imshow(denormalize(img))
        axes[1].imshow(cam, cmap="jet", alpha=0.5)
        axes[1].set_title(f"Grad-CAM (pred={class_names[pred_label]})")
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig("gradcam_example.png", dpi=150)
        print("Đã lưu gradcam_example.png")


if __name__ == "__main__":
    main()
