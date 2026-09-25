# """"""
# 1. FakeDetectorCNN: baseline CNN 3-block
# 2. FakeDetectorDualBranch: CNN + nhánh phân tích tần số (FFT magnitude spectrum),
#    vì artifact của ảnh AI-generated (GAN/diffusion) thường lộ rõ ở high-frequency
#    domain hơn là ở pixel space thường.

# Lý do không chọn ViT / ResNet lớn: ảnh 32x32 quá nhỏ, patch-based transformer
# không đủ context, còn ResNet50 dễ overfit và lãng phí compute cho input bé.
# """

import torch
import torch.nn as nn
import torch.fft


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class FakeDetectorCNN(nn.Module):


    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    # 32x32 -> 16x16
            ConvBlock(32, 64),   # 16x16 -> 8x8
            ConvBlock(64, 128),  # 8x8 -> 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class FrequencyBranch(nn.Module):

    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(32, out_dim)

    def forward(self, x):
        # x: (B, 3, H, W) ảnh gốc đã normalize -> tính FFT trên từng channel
        fft = torch.fft.fft2(x, norm="ortho")
        magnitude = torch.log(torch.abs(fft) + 1e-8)  # log-scale để ổn định
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))  # đưa DC về giữa

        feat = self.conv(magnitude)
        feat = feat.flatten(1)
        return self.proj(feat)


class FakeDetectorDualBranch(nn.Module):
    # CNN + frequency-domain branch

    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.spatial = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.freq_branch = FrequencyBranch(out_dim=64)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128 + 64, num_classes),
        )

    def forward(self, x):
        spatial_feat = self.spatial(x)
        spatial_feat = self.spatial_pool(spatial_feat).flatten(1)

        freq_feat = self.freq_branch(x)

        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(combined)


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)

    m1 = FakeDetectorCNN()
    print("Baseline CNN output:", m1(x).shape)

    m2 = FakeDetectorDualBranch()
    print("Dual-branch output:", m2(x).shape)

    n_params_1 = sum(p.numel() for p in m1.parameters())
    n_params_2 = sum(p.numel() for p in m2.parameters())
    print(f"Params baseline: {n_params_1:,} | Params dual-branch: {n_params_2:,}")
