import os

import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms


class CASIADataset(data.Dataset):
    def __init__(self, data_folder, input_shape=(1, 128, 128)):
        # 加载数据集图片列表
        peoples = os.listdir(data_folder)
        data_list = []
        cnt = 0
        for people in peoples:
            pictures = os.listdir(data_folder + people)
            for picture in pictures:
                picture_path = data_folder + people + "/" + picture
                data_list.append([picture_path, cnt])
            if cnt == 1208:
                break
            else:
                cnt += 1
        self.data_list = np.random.permutation(data_list)
        # 创建图片转换操作
        self.transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomCrop(input_shape[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        item = self.data_list[index]
        image = Image.open(item[0]).convert('L')
        image = self.transforms(image)
        label = np.int32(item[1])
        return image.float(), label

    def __len__(self):
        return len(self.data_list)

    @classmethod
    def load_data(cls, path, batch_size):
        return DataLoader(cls(path), batch_size=batch_size, shuffle=True, num_workers=4)


if __name__ == '__main__':
    from environment_setting import train_dataset_path

    data = CASIADataset(train_dataset_path)
    print(len(data.data_list))
