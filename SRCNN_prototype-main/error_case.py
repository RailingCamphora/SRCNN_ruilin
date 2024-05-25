from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# 打开图像并转换为灰度图像
image = Image.open('./picture/选做作业三图像/cameraman_64X64.png').convert('L')

# 定义 ToTensor 转换
to_tensor = transforms.ToTensor()

# 将图像转换为张量
image_tensor = to_tensor(image)
image_array=np.array(image)
# 查看张量的类型和范围
print(image_array.dtype)
print(image_tensor.dtype)  # 输出: torch.float32
print(torch.min(image_tensor), torch.max(image_tensor))  # 输出: tensor(0.) tensor(1.)
