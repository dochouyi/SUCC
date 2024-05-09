import os
import scipy.io as sio
from datasets.prepare.utils.basePrepare import BasePrepare
import numpy as np
from visualize.Visualize import Visualize_class

class AvccPrepare(BasePrepare):
    def __init__(self, basepath):
        images_path = os.path.join(os.path.abspath(basepath),'images/')
        gts_path = os.path.join(os.path.abspath(basepath),'ground-truth/')
        super(AvccPrepare, self).__init__(img_path=images_path,gts_path=gts_path,ext='*.jpg',is_multi=True, adj_num=3,zoom_para=0.35,down_sample_rate=8)


    def load_gt(self, gt_file_path):
        locations=[]
        with open(gt_file_path, 'r') as f:
            datas=f.readlines()
            for data in datas:
                data=data.strip()
                itmes=data.split(',')
                locations.append([int(itmes[1]),int(itmes[0])])
        locations=np.array(locations)
        return locations

    def get_gtfile_path(self,image_file_path, gt_file_base_path):
        i = image_file_path.index('images/') + 7
        j = image_file_path.index('.jpg')
        substr = image_file_path[i:j]
        gt_file_path = gt_file_base_path + substr + '.txt'
        return gt_file_path

    def get_file_identifier(self,image_file_name):
        i = image_file_name.index('images/') + 7
        j = image_file_name.index('.jpg')
        substr = image_file_name[i:j]
        return substr


def generate():
    pre=AvccPrepare('/home/houyi/Code/activevcc/gt_data/')
    pre.process()


def load_gt(gt_file_path):
    locations=[]
    with open(gt_file_path, 'r') as f:
        datas=f.readlines()
        for data in datas:
            data=data.strip()
            itmes=data.split(',')
            locations.append([int(itmes[1]),int(itmes[0])])
    locations=np.array(locations)
    return locations


def vis():
    # load_gt_func,max_sigma,sigma_rate):

    a=Visualize_class(load_gt,max_sigma=100,sigma_rate=0.35)
    a.draw_head_adaptive_theta_circle(image_file='/home/houyi/Code/activevcc/gt_data/images/0.jpg', gt_file="/home/houyi/Code/activevcc/gt_data/ground-truth/0.txt")
if __name__=="__main__":
    generate()











