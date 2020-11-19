import os
from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import StepLR

import dataset
from environment import device, train_checkpoint_path
from model import FocalLoss, ArcMarginProduct, resnet_face18


@dataclass
class HyperParameter:
    batch_size: int
    max_epoch: int
    checkpoint_save_interval: int
    message: str


class NetTrainer:
    def __init__(self, dataset, hyper_parameter: HyperParameter) -> None:
        self.dataloader = dataset
        self._load_model()
        self._load_criterion()
        self._load_optimizer()
        self._load_scheduler()
        self.epoch = 1
        self.hyper_parameter = hyper_parameter

    def _load_model(self) -> None:
        self.backbone = resnet_face18(use_se=False)
        self.backbone.to(device)
        self.metric = ArcMarginProduct(512, len(self.dataloader.dataset), s=30, m=0.5, easy_margin=False)
        self.metric.to(device)

    def _load_criterion(self) -> None:
        self.criterion = FocalLoss(gamma=2)

    def _load_optimizer(self) -> None:
        self.optimizer = torch.optim.SGD(
            [{'params': self.backbone.parameters()}, {'params': self.metric.parameters()}], lr=1e-1,
            weight_decay=5e-4)

    def _load_scheduler(self) -> None:
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.1)

    def start_train(self) -> None:
        while self.epoch <= self.hyper_parameter.max_epoch:
            self.backbone.train()
            for j, (data_input, label) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                # 注入数据
                data_input = data_input.to(device)
                label = label.to(device).long()
                # 模型计算
                feature = self.backbone(data_input)
                output = self.metric(feature, label)
                loss = self.criterion(output, label)
                # 反向传播优化
                loss.backward()
                self.optimizer.step()
                print(f'epoch {self.epoch}, iter {j} / {len(self.dataloader)}, loss = {loss}')

            self.scheduler.step()
            if self.epoch % self.hyper_parameter.checkpoint_save_interval == 0:
                self.save_checkpoint()
            self.epoch += 1

    def load_checkpoint(self, epoch: int) -> None:
        path = train_checkpoint_path.format(epoch, self.hyper_parameter.message)
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            self.backbone.load_state_dict(checkpoint['backbone'])
            self.metric.load_state_dict(checkpoint['metric'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch = checkpoint['epoch'] + 1
            print(f'加载参数：{path}')

    def save_checkpoint(self) -> None:
        path = train_checkpoint_path.format(self.epoch, self.hyper_parameter.message)
        torch.save({
            'backbone': self.backbone.state_dict(),
            'metric': self.metric.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
        }, path)
        print(f'保存参数：{path}')


if __name__ == '__main__':
    # 模型超参数
    hyper_parameter = HyperParameter(
        batch_size=16,
        max_epoch=30,
        checkpoint_save_interval=1,
        message='casia'
    )
    # 加载训练数据
    train_dataloader = dataset.get_casia_dataloader(hyper_parameter.batch_size)

    # 加载模型
    net_trainer = NetTrainer(train_dataloader, hyper_parameter)

    # 加载参数
    # net_trainer.load_checkpoint(epoch=1)

    # 开始训练
    net_trainer.start_train()
