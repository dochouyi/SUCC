import os
from datasets.prepare.utils.basePrepare import BasePrepare
import numpy as np

class CrowdXPrepare(BasePrepare):
    def __init__(self,basepath,):

        images_path = os.path.join(os.path.abspath(basepath),'images/')
        gts_path = os.path.join(os.path.abspath(basepath),'ground-truth/')
        super(CrowdXPrepare, self).__init__(img_path=images_path,gts_path=gts_path,ext='*.png',is_multi=True, adj_num=3,zoom_para=0.3,down_sample_rate=8)


    def load_gt(self, gt_file_path):
        locations=[]
        with open(gt_file_path, 'r') as f:
            datas=f.readlines()
            for data in datas:
                data=data.strip()
                itmes=data.split(',')
                locations.append([int(itmes[0]),int(itmes[1])])
        locations=np.array(locations)
        return locations

    def get_gtfile_path(self,image_file_path, gt_file_base_path):
        i = image_file_path.index('images/') + 7
        j = image_file_path.index('.png')
        substr = image_file_path[i:j]
        gt_file_path = gt_file_base_path + substr + '.txt'
        return gt_file_path


    def get_file_identifier(self,image_file_name):
        i = image_file_name.index('images/') + 7
        j = image_file_name.index('.png')
        substr = image_file_name[i:j]
        return substr


def generate():
    pre=CrowdXPrepare('/home/houyi/datasets/CrowdX_Video/train_data')
    pre.process()

    # pre=CrowdXPrepare('/home/houyi/datasets/CrowdX_Video/test_data')
    # pre.process()


if __name__=="__main__":
    generate()











