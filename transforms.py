"""
transforms.py
QUAN TRỌNG: File này phải được import CHUNG bởi cả train.py và inference/API.
Bài học từ project traffic sign: nếu train và inference dùng transform khác nhau
(ví dụ khác mean/std, khác resize) -> model sẽ predict sai một cách âm thầm,
không báo lỗi gì cả, rất khó debug.
"""

from torchvision import transforms


MEAN = [0.4719943106174469, 0.46289336681365967, 0.41781511902809143]  
STD  = [0.23757432401180267, 0.2374235838651657, 0.2659563422203064]    

IMG_SIZE = 32

# ---- Transform cho training: có augmentation ----
# Lưu ý: KHÔNG dùng augment phá tần số cao (blur mạnh, JPEG random quality)
# vì fake-image artifact thường nằm ở high-frequency, augment mạnh có thể xóa mất.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(IMG_SIZE, padding=2, padding_mode="reflect"),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # nhẹ thôi
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ---- Transform cho validation/test/inference: KHÔNG augmentation ----
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def compute_mean_std(dataset_root: str):
    """
    Chạy 1 lần trên tập train để lấy mean/std thật, sau đó paste kết quả
    vào MEAN, STD ở trên.

    Usage:
        python -c "from transforms import compute_mean_std; compute_mean_std('data/train')"
    """
    import torch
    from torchvision import datasets
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    ds = datasets.ImageFolder(dataset_root, transform=transforms.ToTensor())
    print(f"Tổng số ảnh: {len(ds)}", flush=True)

    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0

    for images, _ in tqdm(loader, desc="Đang tính mean/std"):
        b, c, h, w = images.shape
        mean += images.sum(dim=[0, 2, 3])
        std += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels += b * h * w

    mean /= n_pixels
    std = (std / n_pixels - mean ** 2).sqrt()

    print(f"MEAN = {mean.tolist()}")
    print(f"STD  = {std.tolist()}")
    return mean.tolist(), std.tolist()