import os
import glob
from PIL import Image
from datasets.prepare.utils.gaussianGenerator import GaussianGenerator
from datasets.prepare.utils.resizeUtil import ResizeUtil
from datasets.prepare.utils.fileListProcess import FileListProcess
from abc import ABCMeta,abstractmethod
import numpy as np


#这个准备操作不会做数据增强处理，需要注意
class BasePrepare(metaclass=ABCMeta):
    def __init__(self,img_path,gts_path, ext='*.jpg',is_multi=False,adj_num=3,zoom_para=0.3,down_sample_rate=8):
        '''
        :param img_path(str):   例如'/home/houyi/datasets/ShanghaiTech/part_B/train_data/images/'
        :param gts_path(str):   例如'/home/houyi/datasets/ShanghaiTech/part_B/train_data/ground-truth/'
        :param ext(str):
        :param is_multi(bool):
        :param adj_num(int):
        :param zoom_para(float):
        :param down_sample_rate(int):
        '''
        #分别是2个路径
        self.img_path = img_path
        self.gts_path = gts_path
        #获取所有图像组成的列表
        self.image_files=sorted(glob.glob(os.path.join(self.img_path, ext)))    #按照什么样的顺序处理其实问题都不大，所以这里就暂时这样，在这里不会考虑顺序，是按照字典顺序排列的

        self.generator=GaussianGenerator(adj_num=adj_num, max_sigma=100, zoom_para=zoom_para)
        self.resizeUtil=ResizeUtil(down_sample_rate=down_sample_rate)
        self.fileListProcess=FileListProcess(self.process_singlefile,self.image_files,is_multi=is_multi)

        dirpath=os.path.abspath(os.path.join(os.path.abspath(self.img_path),'..'))
        self.saved_npy_path=os.path.join(dirpath,'ADJ{a}_ZOOM{z}_DR{d}'.format(a=adj_num,z=zoom_para,d=down_sample_rate))
        #如果没有路径生成，则进行生成
        if not os.path.exists(self.saved_npy_path):
            os.makedirs(self.saved_npy_path)
    @abstractmethod
    def load_gt(self, gt_file_path):
        '''
        :param gt_file_path(str):
        :return(ndarray(point_num,2)):
        '''
        pass
    @abstractmethod
    def get_gtfile_path(self,image_file_path, gt_file_base_path):
        pass
    @abstractmethod
    def get_file_identifier(self,image_file_name):
        pass

    def process_singlefile(self,image_file_path):
        '''
        :param image_file_path(str):img path
        :return:
        '''
        print(image_file_path)
        image = Image.open(image_file_path, mode='r').convert("RGB")
        gt_file_path=self.get_gtfile_path(image_file_path,self.gts_path)
        locations = self.load_gt(gt_file_path)
        image, desity_map, dot_map=self.generator.process(image,locations)
        npy_name = os.path.join(self.saved_npy_path,self.get_file_identifier(image_file_path) + '.npy')
        _desity_map=self.resizeUtil.resize_density(desity_map)
        _dot_map=self.resizeUtil.resize_dotmap(dot_map)
        np.save(npy_name, (image,_desity_map,_dot_map))#此处的image为ndarray格式

    def process(self):
        self.fileListProcess.process()