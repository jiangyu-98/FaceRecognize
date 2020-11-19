import os

import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

from environment import cv2


class Dataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = np.random.permutation(data_list)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        image = cv2.imread(image_path)
        image = self.transforms(image)
        return image.float(), np.int32(label)

    def __len__(self):
        return len(self.data_list)

    @classmethod
    def load_data(cls, path, batch_size):
        return DataLoader(cls(path), batch_size=batch_size, shuffle=True, num_workers=4)


def get_lfw_datalist(condition=lambda x: x.endswith('.png')):
    from environment import lfw_path as data_folder
    data_list = []
    for cnt, people in enumerate(os.listdir(data_folder)):
        if os.path.isfile(data_folder + people):
            continue
        for picture in filter(condition, os.listdir(data_folder + people)):
            data_list.append([data_folder + people + '/' + picture, cnt])
    return data_list


def get_casia_datalist(condition=lambda x: x.endswith('.png')):
    from environment import casia_path as data_folder
    data_list = []
    for cnt, people in enumerate(os.listdir(data_folder)):
        if os.path.isfile(data_folder + people):
            continue
        for idx, picture in enumerate(filter(condition, os.listdir(data_folder + people))):
            data_list.append([data_folder + people + '/' + picture, cnt])
            if idx > 10:
                break
        if len(data_list) > 50000:
            break
    return data_list


def get_mask_slim_datalist(condition=lambda x: x.endswith('.png')):
    from environment import mask_slim_path as data_folder
    data_list = []
    for cnt, people in enumerate(os.listdir(data_folder)):
        if os.path.isfile(data_folder + people):
            continue
        for idx, picture in enumerate(filter(condition, os.listdir(data_folder + people))):
            data_list.append([data_folder + people + '/' + picture, cnt])
    return data_list


def get_lfw_dataloader(batch_size):
    data_list = get_lfw_datalist()
    return DataLoader(Dataset(data_list), batch_size=batch_size, shuffle=True, num_workers=4)


def get_mask_slim_dataloader(batch_size):
    data_list = get_mask_slim_datalist()
    return DataLoader(Dataset(data_list), batch_size=batch_size, shuffle=True, num_workers=4)


def get_casia_dataloader(batch_size):
    data_list = get_casia_datalist()
    return DataLoader(Dataset(data_list), batch_size=batch_size, shuffle=True, num_workers=4)


def align_dataset(datalist_generator):
    from face_align import FaceAligner

    data_list = datalist_generator(lambda x: x.endswith('.jpg'))
    face_aligner = FaceAligner()
    for picture, _ in data_list:
        try:
            if os.path.exists(picture.replace('jpg', 'png')):
                print(f'exist {picture}')
                continue
            image = cv2.imread(picture)
            image = face_aligner.align(image)
            cv2.imwrite(picture.replace('jpg', 'png'), image)
            print(f'transform {picture} finished!')
        except Exception as e:
            print(f'transform {picture} failed! \n {e}')


if __name__ == '__main__':
    # v = get_casia_datalist()
    # print(v)
    # print(len(v))

    print(align_dataset(get_casia_datalist))
