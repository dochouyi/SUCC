import os
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
import numpy as np
import glob
from PIL import Image

class CrowdDataset(Dataset):
    def __init__(self, saved_npy_path):

        self.saved_npy_path = saved_npy_path
        self.npy_files = sorted(glob.glob(os.path.join(self.saved_npy_path, '*.npy')))
        self.img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.density_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
        ])
        self.dot_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        file_path = self.npy_files[index]
        data_item = np.load(file_path,allow_pickle=True)
        img = Image.fromarray(data_item[0])
        density_map = Image.fromarray(data_item[1])
        dot_map = Image.fromarray(data_item[2])
        img = self.img_transform(img)
        density_map = self.density_transform(density_map)
        dot_map = self.dot_transform(dot_map)
        return img, density_map, dot_map

    def __len__(self):
        return len(self.npy_files)