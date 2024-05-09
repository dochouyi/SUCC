import os
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
import numpy as np
import glob
from torch.utils.data import DataLoader
from PIL import Image
import torch

class CrowdVideoDataset(Dataset):
    def __init__(self, saved_npy_path, seq_len=5, train_flag=True):

        self.saved_npy_path=saved_npy_path
        npy_files = sorted(glob.glob(os.path.join(self.saved_npy_path, '*.npy')))

        self.video_num=len(npy_files)//seq_len
        self.train_flag=train_flag

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
        if self.train_flag==True:
            seq_index=index // 9
            inner_index=index % 9
            seq_npy_name=str(seq_index)+'_'+str(inner_index)+'_*.npy'
        else:
            seq_npy_name=str(index)+'_*.npy'
        seq_npy_files = sorted(glob.glob(os.path.join(self.saved_npy_path, seq_npy_name)))
        frame_imgs= []
        frame_densitys= []
        frame_dot_maps= []
        for file_path in seq_npy_files:
            data_item = np.load(file_path,allow_pickle=True)
            img = Image.fromarray(data_item[0])
            density_map = Image.fromarray(data_item[1])
            dot_map = Image.fromarray(data_item[2])

            img = self.img_transform(img)
            density_map = self.density_transform(density_map)
            dot_map = self.dot_transform(dot_map)
            frame_imgs.append(img)
            frame_densitys.append(density_map)
            frame_dot_maps.append(dot_map)
        frame_imgs = torch.stack(frame_imgs, dim=0)
        frame_densitys = torch.stack(frame_densitys, dim=0)
        frame_dot_maps = torch.stack(frame_dot_maps, dim=0)
        return frame_imgs, frame_densitys, frame_dot_maps

    def __len__(self):
        return self.video_num


class loader_wrapper(object):
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for imgs, densitys, dot_maps in self.loader:
            yield imgs.permute(1, 0, 2, 3, 4), densitys.permute(1, 0, 2, 3, 4), dot_maps.permute(1, 0, 2, 3, 4)

    def __len__(self):
        return len(self.loader)



if __name__=="__main__":

    train_dataset=CrowdVideoDataset('/home/houyi/datasets/CrowdX_Video/train_data/aug_val_data',train_flag=True)
    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=10, shuffle=True, drop_last=True)

    for data, target,target2 in train_loader:
        batchSize = data.size(0)
        pass





