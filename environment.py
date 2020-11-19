import os

import cv2
import torch

__all__ = [
    'device',
    'cv2',
    'project_path',
    'mask_slim_path',
    'lfw_path',
    'casia_path',
    'train_checkpoint_path',
    'test_dataset_path',
    'test_model_weight_path',
    'run_model_weight_path',
    'run_fingerprint_database_path',
]

# 计算设备
if "TORCH_DEVICE" in os.environ:
    device = torch.device(os.environ['TORCH_DEVICE'])
else:
    device = torch.device('cpu')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 项目根目录
project_path = './'
dataset_path = '../dataset/'
if not os.path.exists(project_path + "checkpoint"):
    os.mkdir(project_path + "checkpoint")
# 数据集
mask_slim_path = dataset_path + 'test-mask-slim/'
lfw_path = dataset_path + 'lfw/'
casia_path = dataset_path + 'CASIA-WebFace/'

# 训练后保存的模型参数位置格式
train_checkpoint_path = project_path + 'checkpoint/checkpoint_epoch_{}_{}.pth'

# 测试数据集目录
test_dataset_path = dataset_path + 'lfw/'
test_model_weight_path = project_path + 'checkpoint/checkpoint_epoch_13_lfw.pth'

# 运行模型参数路径
run_model_weight_path = project_path + 'checkpoint/checkpoint_epoch_13_lfw.pth'
# 指纹数据库路径
run_fingerprint_database_path = project_path + 'fingerprint.db'
