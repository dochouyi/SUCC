from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms


class ImgDataset(Dataset):
    def __init__(self,imgs):
        '''
        :param imgs(list(PIL.Image)):
        '''
        self.imgs=imgs
        self.img_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    def __getitem__(self, index):
        img = self.imgs[index]

        img = self.img_transform(img)

        return img

    def __len__(self):
        return len(self.imgs)

