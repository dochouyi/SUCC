import os
import scipy.io as sio
from datasets.prepare.utils.basePrepare import BasePrepare


class ShhaiPrepare_B(BasePrepare):
    def __init__(self, basepath):
        images_path = os.path.join(os.path.abspath(basepath),'images/')
        gts_path = os.path.join(os.path.abspath(basepath),'ground-truth/')
        super(ShhaiPrepare_B, self).__init__(img_path=images_path,gts_path=gts_path,ext='*.jpg',is_multi=True, adj_num=3,zoom_para=0.3,down_sample_rate=8)


    def load_gt(self, gt_file_path):
        '''
        :param gt_file_path(str):
        :return(ndarray(point_num,2)):
        '''
        dataset = sio.loadmat(gt_file_path)
        dataset = dataset['image_info'][0, 0]
        locations = dataset['location'][0, 0]
        return locations

    def get_gtfile_path(self,image_file_path, gt_file_base_path):
        i = image_file_path.index('IMG_') + 4
        j = image_file_path.index('.jpg')
        substr = image_file_path[i:j]
        gt_file_path = gt_file_base_path + 'GT_IMG_' + substr + '.mat'
        return gt_file_path

    def get_file_identifier(self,image_file_name):
        i = image_file_name.index('IMG_') + 4
        j = image_file_name.index('.jpg')
        substr = image_file_name[i:j]
        return substr


def generate():
    pre=ShhaiPrepare_B('/home/houyi/datasets/ShanghaiTech/part_B/train_data')
    pre.process()
    # pre=ShhaiPrepare_B('/home/houyi/datasets/ShanghaiTech/part_B/test_data')
    # pre.process()

if __name__=="__main__":
    generate()











