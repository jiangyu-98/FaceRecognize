import os

import torch

__all__ = [
    'device',
    'project_path',
    'train_checkpoint_path',
    'train_dataset',
    'train_dataset_path',
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

# 训练数据集
from datasetloader import LFWDataset

train_dataset = LFWDataset
train_dataset_path = dataset_path + 'lfw-align-128/'
# from datasetloader import CASIADataset
# train_dataset_path = dataset_path + 'CASIA-WebFace/'
# train_dataset = CASIADataset
# 训练后保存的模型参数位置格式
train_checkpoint_path = project_path + 'checkpoint/checkpoint_epoch_{}_{}.pth'

# 测试数据集目录
test_dataset_path = dataset_path + 'lfw-align-128/'
test_model_weight_path = project_path + 'checkpoint/checkpoint_epoch_10.pth'

# 运行模型参数路径
run_model_weight_path = project_path + 'checkpoint/checkpoint_epoch_8_lfw.pth'
# 指纹数据库路径
run_fingerprint_database_path = project_path + 'fingerprint.db'
