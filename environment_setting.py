import os

import torch

from datasetloader import LFWDataset

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

# 项目根目录
project_path = 'D:/Projects/大规模多媒体信息检索与管理/FaceRecognize/'
dataset_pah = 'D:/Projects/大规模多媒体信息检索与管理/dataset/'

# 训练数据集目录
train_dataset_path = dataset_pah + '../dataset/lfw-align-128/'
# 测试集的数据加载器
train_dataset = LFWDataset
# 训练后保存的模型参数位置格式
train_checkpoint_path = project_path + 'checkpoint/checkpoint_epoch_{}{}.pth'

# 测试数据集目录
test_dataset_path = dataset_pah + 'dataset/lfw-align-128/'
test_model_weight_path = project_path + 'checkpoint/checkpoint_epoch_3.pth'

# 运行模型参数路径
run_model_weight_path = project_path + 'checkpoint/checkpoint_epoch_8.pth'
# 指纹数据库路径
run_fingerprint_database_path = project_path + 'fingerprint.db'
