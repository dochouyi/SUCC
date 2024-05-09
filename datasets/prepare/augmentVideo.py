import os
import glob
import numpy as np
from datasets.prepare.utils.augmentUtils import video_generate_slices


class AugmentVideo():
    '''
    对视频数据训练集的增强体现在文件命名由0_0->0_(0-8)_0
    '''
    def __init__(self,saved_npy_path='',seq_len=5):
        '''
        :param saved_npy_path:npy文件的路径
        :param seq_len:每个序列的长度
        '''
        self.saved_npy_path=saved_npy_path
        self.npy_files=sorted(glob.glob(os.path.join(self.saved_npy_path, '*.npy')))
        self.seq_len=seq_len

        dirpath=os.path.abspath(os.path.join(saved_npy_path,'..'))
        pre_fix=os.path.basename(saved_npy_path)
        self.saved_augment_npy_path=os.path.join(dirpath,pre_fix+'_AUG{num}'.format(num=18))
        if not os.path.exists(self.saved_augment_npy_path):
            os.makedirs(self.saved_augment_npy_path)

        self.video_num=len(self.npy_files)//self.seq_len    #有多少个视频文件

    def process(self):
        for index in range(self.video_num):
            images=[]
            densitys=[]
            dot_maps=[]
            for i in range(self.seq_len):
                file_name=str(index)+'_'+str(i)+'.npy'
                file_path=os.path.join(self.saved_npy_path,file_name)
                data_item = np.load(file_path,allow_pickle=True)
                images.append(data_item[0])
                densitys.append(data_item[1])
                dot_maps.append(data_item[2])
            out_images, out_density_maps,out_dot_maps,indexes=video_generate_slices(images,densitys,dot_maps)
            for i,img in enumerate(out_images):
                saved_npy_file_path=os.path.join(self.saved_augment_npy_path,str(index)+'_'+str(indexes[i][0])+'_'+str(indexes[i][1])+'.npy')
                np.save(saved_npy_file_path, (out_images[i],out_density_maps[i],out_dot_maps[i]))


def my_test_Video_Prepare():

    # aug=AugmentVideo('/home/houyi/datasets/CrowdX_Video/train_data/temp')
    # aug.process()

    aug=AugmentVideo('/home/houyi/datasets/CrowdX_Video/train_data/ADJ3_ZOOM0.3_DR8')
    aug.process()


if __name__=="__main__":
    my_test_Video_Prepare()