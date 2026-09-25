
from torchvision import transforms


MEAN = [0.4719943106174469, 0.46289336681365967, 0.41781511902809143]  
STD  = [0.23757432401180267, 0.2374235838651657, 0.2659563422203064]    

IMG_SIZE = 32

# ---- Transform cho training: có augmentation ----
# KHÔNG dùng augment phá tần số cao (blur mạnh, JPEG random quality)
# vì fake-image artifact thường nằm ở high-frequency, augment mạnh có thể xóa mất.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(IMG_SIZE, padding=2, padding_mode="reflect"),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ---- Transform cho validation/test/inference: KHÔNG augmentation ----
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def compute_mean_std(dataset_root: str):
    
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