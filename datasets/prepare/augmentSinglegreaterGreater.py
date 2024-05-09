
import glob
import os
import numpy as np
from datasets.prepare.utils.augmentUtils import generate_slices,flip_slices


class AugmentSingleGreater():
    def __init__(self,saved_npy_path=''):
        '''
        输入是经过预处理的npy的根路径，然后就在同样的路径生成_AUG{num}结尾的切片增强版本所有的文件
        :param saved_npy_path:
        '''
        saved_npy_path=os.path.abspath(saved_npy_path)
        self.npy_files=sorted(glob.glob(os.path.join(saved_npy_path, '*.npy')))
        dirpath=os.path.abspath(os.path.join(saved_npy_path,'..'))
        pre_fix=os.path.basename(saved_npy_path)
        self.saved_augment_npy_path=os.path.join(dirpath,pre_fix+'_AUG{num}'.format(num=18))
        if not os.path.exists(self.saved_augment_npy_path):
            os.makedirs(self.saved_augment_npy_path)
    def process(self,):
        '''
        因为是每一张图单独保存，所以数据集中图像的大小不一样的话也是可以处理的
        :return:
        '''
        for file_path in self.npy_files:
            file_prefix=os.path.splitext(os.path.basename(file_path))[0]
            data_item = np.load(file_path,allow_pickle=True)
            image=data_item[0]
            density_map=data_item[1]
            dot_map=data_item[2]
            images, density_maps, dot_maps = generate_slices(image, density_map, dot_map)
            images, density_maps, dot_maps = flip_slices(images, density_maps, dot_maps)
            for i,img in enumerate(images):
                npy_name=os.path.join(self.saved_augment_npy_path,file_prefix+'_'+str(i) + '.npy')
                if np.sum(density_maps[i])<1:
                    continue
                np.save(npy_name, (img,density_maps[i],dot_maps[i]))
        print('Done!')

def generate():
    pre=AugmentSingleGreater('/home/houyi/Code/activevcc/gt_data/ADJ3_ZOOM0.35_DR8')
    pre.process()


if __name__=="__main__":
    generate()
