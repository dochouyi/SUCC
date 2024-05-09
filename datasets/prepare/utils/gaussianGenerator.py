import math
import scipy
from scipy import ndimage
import scipy.spatial
import numpy as np


class GaussianGenerator():
    def __init__(self, adj_num=3, max_sigma=100, zoom_para=0.3):
        '''
        :param adj_num(int):每计算自适应高斯核临接点数目，例如adj_num=3的情况下，邻居数目是3
        :param max_sigma(int):最大sigma的大小，高斯核所对应的sigma的大小
        :param zoom_para(float):计算过sigma之后要做缩放操作的倍数
        :param dot_map_value:点图是uint8类型，人头点所在位置赋予的值，通常为1
        :param pre_gaussian_value:高斯函数中要赋予的值，通常为1
        '''
        self.__adj_num=adj_num+1
        self.__max_sigma=max_sigma
        self.__zoom_para=zoom_para
        self.__dot_map_value=1
        self.__pre_gaussian_value=1.0

    def __dot_map_generator(self, locations, image_shape):
        '''
        :param locations(ndarray(point_num,2)):float64
        :param image_shape(tuple(2)):int
        :return(ndarray(img_height,img_width)):uint8
        '''
        dot_map = np.zeros((image_shape[1], image_shape[0]), dtype='uint8')
        for dot in locations:
            try:
                x = int(math.floor(dot[1]))
                y = int(math.floor(dot[0]))
                dot_map[x, y] = self.__dot_map_value
            except IndexError:
                print((image_shape[1], image_shape[0]))
        return dot_map

    def __gaussian_generator(self,dot_map):
        '''
        :param dot_map(ndarray(img_height,img_width)):uint8
        :return(ndarray(img_height,img_width)):float32
        '''
        density_map = np.zeros(dot_map.shape, dtype=np.float32)
        gt_count = np.count_nonzero(dot_map)
        if gt_count == 0:
            return density_map
        pts = np.array(list(zip(np.nonzero(dot_map)[0], np.nonzero(dot_map)[1])))
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=2048)
        distances, _ = tree.query(pts, k=self.__adj_num)
        for i, pt in enumerate(pts):
            pt2d = np.zeros(dot_map.shape, dtype=np.float32)
            pt2d[pt[0], pt[1]] = self.__pre_gaussian_value
            if gt_count > 1:
                sigma = np.average(distances[i][1:])
                if sigma > self.__max_sigma:
                    sigma = self.__max_sigma
            else:
                sigma = self.__max_sigma
            sigma = sigma * self.__zoom_para
            density_map += ndimage.filters.gaussian_filter(pt2d, sigma)
        return density_map

    def process(self, image, locations):
        '''
        :param image(PIL.Image):RGB格式的Image类型
        :param locations(ndarray(point_num,2)):float64，每个人头点的位置（x,y）,x是距离图像左边的距离，y是距离图像上边的距离
        :return[ndarray(img_height,img_width,3):uint8,ndarray(img_height,img_width):float32,ndarray(img_height,img_width):uint8]
        '''
        dot_map = self.__dot_map_generator(locations, image.size)
        desity_map = self.__gaussian_generator(dot_map)
        img=np.array(image)
        return img, desity_map, dot_map


