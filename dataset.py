
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from collections import Counter

from transforms import train_transform, eval_transform


def get_dataloaders(data_root: str='G:/ML/files/Fake_vs_Real_Photo/train', 
                    batch_size: int = 128, # the number of images in each batch
                    num_workers: int = 2, # rely on CPU , num_workers is sub processes created by python for missions: read images->decode->transform->tensor
                    handle_imbalance: bool = True): #handle data imbalance, defaul is True
    """
    """
    train_ds = datasets.ImageFolder(f"{data_root}/train", transform=train_transform)
    test_ds = datasets.ImageFolder(f"{data_root}/test", transform=eval_transform)

    class_names = train_ds.classes  # for example ['fake', 'real'] - following alphabet
    print(f"Classes: {class_names}") # print list of classes
    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}") #print the total images of train and test 

    if handle_imbalance:
        label_counts = Counter([label for _, label in train_ds.samples]) 
        print(f"Phân bố lớp trong train: {label_counts}")

        # weight tỉ lệ nghịch với tần suất -> lớp ít mẫu được sample nhiều hơn
        class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
        sample_weights = [class_weights[label] for _, label in train_ds.samples]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                         replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, class_names


if __name__ == "__main__":
    # Quick sanity check
    train_loader, test_loader, class_names = get_dataloaders("Fake_vs_Real_Photo")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, labels: {labels[:10]}")

# #Classes: ['FAKE', 'REAL']
# Train size: 100000, Test size: 20000
# Phân bố lớp trong train: Counter({0: 50000, 1: 50000})
# Batch shape: torch.Size([128, 3, 32, 32]), labels: tensor([1, 0, 1, 0, 1, 1, 0, 0, 0, 0])