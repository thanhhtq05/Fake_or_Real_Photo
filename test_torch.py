import torch

print(f"PyTorch Version: {torch.__version__}")

# Kiểm tra xem PyTorch có nhận diện được GPU không
if torch.cuda.is_available():
    print(f"GPU khả dụng: {torch.cuda.get_device_name(0)}")
else:
    print("Đang chạy trên CPU")

# Tạo thử một Tensor đơn giản
x = torch.rand(2, 3)
print("Tensor ngẫu nhiên:\n", x)