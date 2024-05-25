import os
import shutil

# 定义路径
source_dir = 'D:/DIV2K_train_LR_bicubic_X2/DIV2K_train_LR_bicubic/X2'
train_dir = './dataset/BSDS300/images/train'
test_dir = './dataset/BSDS300/images/test'

# 获取所有图片文件的列表
image_files = sorted(os.listdir(source_dir))

# 确保图片文件夹中至少有800张图片
if len(image_files) < 800:
    raise ValueError("The source directory does not contain at least 800 images.")

# 前700张加入训练集
for image_file in image_files[:700]:
    shutil.move(os.path.join(source_dir, image_file), os.path.join(train_dir, image_file))

# 后100张加入测试集
for image_file in image_files[700:800]:
    shutil.move(os.path.join(source_dir, image_file), os.path.join(test_dir, image_file))

print("Images have been successfully moved.")