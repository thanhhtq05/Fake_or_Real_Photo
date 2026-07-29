"""
evaluate.py — BƯỚC 5 (Đánh giá mô hình)

Chạy: python evaluate.py --checkpoint best_model.pt --data-root data
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)

from dataset import get_dataloaders
from model import FakeDetectorCNN, FakeDetectorDualBranch


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]  # xác suất lớp "real" (index 1 nếu alphabetical)
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_names = ckpt["class_names"]

    model_cls = FakeDetectorCNN if ckpt["model_name"] == "baseline" else FakeDetectorDualBranch
    model = model_cls().to(device)
    model.load_state_dict(ckpt["model_state"])

    _, test_loader, _ = get_dataloaders(args.data_root, batch_size=args.batch_size,
                                         handle_imbalance=False)

    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(f"           Pred:{class_names[0]:>8}  Pred:{class_names[1]:>8}")
    print(f"True:{class_names[0]:>6}  {cm[0][0]:>10}  {cm[0][1]:>13}")
    print(f"True:{class_names[1]:>6}  {cm[1][0]:>10}  {cm[1][1]:>13}")

    auc = roc_auc_score(y_true, y_probs)
    print(f"\nROC-AUC: {auc:.4f}")

    # --- Vẽ ROC curve + PR curve, lưu ra file ảnh ---
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_probs)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_probs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(recall, precision)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")

    plt.tight_layout()
    out_img_name = f"evaluation_curves_{ckpt.get('model_name', 'model')}.png"
    plt.savefig(out_img_name, dpi=150)
    print(f"\nĐã lưu ROC/PR curve vào {out_img_name}")

    # --- Gợi ý threshold tốt nhất theo Youden's J statistic (cân bằng chung 2 lớp) ---
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    print(f"\nThreshold tối ưu (Youden's J, cân bằng chung): {roc_thresholds[best_idx]:.4f} "
          f"(mặc định đang dùng 0.5)")

    # =========================================================================
    # PHÂN TÍCH THRESHOLD THEO MỤC TIÊU: KHÔNG BỎ LỌT ẢNH FAKE
    # (recall của lớp FAKE là ưu tiên số 1, không phải cân bằng chung)
    # =========================================================================
    print("\n" + "=" * 60)
    print("THRESHOLD THEO RECALL-FAKE (ưu tiên bắt hết ảnh fake)")
    print("=" * 60)

    fake_idx = class_names.index("FAKE")
    real_idx = class_names.index("REAL")

    # y_probs hiện tại là P(REAL) (lớp index 1). Đổi sang P(FAKE) = 1 - P(REAL)
    # (chỉ đúng khi bài toán 2 lớp, mà đúng bài này là vậy).
    prob_fake = 1 - y_probs if real_idx == 1 else y_probs
    y_is_fake = (y_true == fake_idx).astype(int)  # 1 nếu ảnh thật sự là FAKE

    precision_fake, recall_fake, pr_thresholds_fake = precision_recall_curve(y_is_fake, prob_fake)

    # Bảng đánh đổi: ứng với mỗi threshold, recall/precision FAKE là bao nhiêu
    print(f"{'Threshold':>10} {'Recall-FAKE':>12} {'Precision-FAKE':>16} {'Ảnh REAL bị báo nhầm':>22}")
    n_real_total = (y_true == real_idx).sum()
    for th in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1]:
        preds_at_th = (prob_fake >= th).astype(int)
        tp = ((preds_at_th == 1) & (y_is_fake == 1)).sum()
        fp = ((preds_at_th == 1) & (y_is_fake == 0)).sum()
        fn = ((preds_at_th == 0) & (y_is_fake == 1)).sum()
        recall_at_th = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_at_th = tp / (tp + fp) if (tp + fp) > 0 else 0
        false_alarm_rate = fp / n_real_total if n_real_total > 0 else 0
        print(f"{th:>10.2f} {recall_at_th:>12.4f} {precision_at_th:>16.4f} "
              f"{fp:>8} / {n_real_total} ({false_alarm_rate:.2%})")

    # Tìm threshold nhỏ nhất đạt được các mức recall mục tiêu cụ thể
    print(f"\n--- Threshold cần dùng để đạt recall-FAKE mục tiêu ---")
    for target_recall in [0.99, 0.995, 0.999]:
        # pr_thresholds_fake có độ dài = len(recall_fake) - 1, recall_fake giảm dần theo threshold tăng
        valid_idx = np.where(recall_fake[:-1] >= target_recall)[0]
        if len(valid_idx) > 0:
            # threshold LỚN NHẤT vẫn đảm bảo recall >= target (giữ precision cao nhất có thể)
            idx = valid_idx[np.argmax(pr_thresholds_fake[valid_idx])]
            th = pr_thresholds_fake[idx]
            prec = precision_fake[idx]
            print(f"  Recall-FAKE >= {target_recall:.1%}: cần threshold ~{th:.4f} "
                  f"(precision-FAKE tương ứng: {prec:.4f})")
        else:
            print(f"  Recall-FAKE >= {target_recall:.1%}: không đạt được ngay cả ở threshold thấp nhất")

    print("\nGợi ý đọc bảng: threshold ở đây là ngưỡng xác suất P(FAKE) để gán nhãn 'FAKE'.")
    print("Threshold càng THẤP -> recall-FAKE càng CAO (bắt được nhiều fake hơn),")
    print("nhưng đổi lại càng nhiều ảnh REAL bị báo nhầm thành FAKE (precision-FAKE giảm).")
    print("Chọn threshold theo mức độ chấp nhận được của false alarm trong ứng dụng thực tế.")


if __name__ == "__main__":
    main()