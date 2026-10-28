import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DerainDataset(Dataset):
    def __init__(self, root_dir, is_train=True, image_size=128):
        self.root_dir = root_dir
        self.is_train = is_train

        data_folder = 'train' if is_train else 'test'
        self.rainy_dir = os.path.join(self.root_dir, data_folder, 'input')
        self.clean_dir = os.path.join(self.root_dir, data_folder, 'target')

        self.rainy_images = sorted(os.listdir(self.rainy_dir))

        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomCrop(image_size),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        clean_path = os.path.join(self.clean_dir, self.rainy_images[idx])

        rainy_image = Image.open(rainy_path).convert('RGB')
        clean_image = Image.open(clean_path).convert('RGB')

        rainy_tensor = self.transform(rainy_image)
        clean_tensor = self.transform(clean_image)

        # 同步水平翻转：只掷一次骰子，翻就两张一起翻
        if self.is_train and random.random() < 0.5:
            rainy_tensor = torch.flip(rainy_tensor, dims=[2])
            clean_tensor = torch.flip(clean_tensor, dims=[2])

        return rainy_tensor, clean_tensor, self.rainy_images[idx]
    