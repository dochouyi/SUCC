
from PIL import Image
import numpy as np


class ResizeUtil():
    def __init__(self, down_sample_rate=8):
        self.down_sample_rate=down_sample_rate

    def resize_density(self, density_map):
        '''
        :param density_map(ndarray(img_height,img_width))-float32: numpy格式,输入原尺寸
        :return(ndarray(img_height//self.down_sample_rate,img_width//self.down_sample_rate)):float32,返回resize之后的密度图,返回的密度图乘上了系数，使得总的人数没有发生变化
        '''
        sum1=np.sum(density_map)
        den = Image.fromarray(density_map)
        x, y = den.size
        temp = den.resize((x // self.down_sample_rate, y // self.down_sample_rate), Image.BILINEAR)
        post_density=np.array(temp)
        sum2=np.sum(post_density)
        if sum2==0:
            return post_density
        return post_density * sum1/sum2

    def resize_dotmap(self, dot_map):
        '''
        :param dot_map(ndarray(img_height,img_width))-uint8:人头点图,numpy格式,输入原尺寸
        :return(ndarray(img_height//self.down_sample_rate,img_width//self.down_sample_rate)):float32,返回numpy格式的人头点图，在有点的地方像素值为1，这里采用直接累加点值的方法，因此有些地方的人头值较高
        '''
        pts = np.array(list(zip(np.nonzero(dot_map)[0], np.nonzero(dot_map)[1])))
        shape=(dot_map.shape[0] // self.down_sample_rate,dot_map.shape[1] // self.down_sample_rate)
        post_dotmap = np.zeros(shape, dtype=np.float32)
        for i, pt in enumerate(pts):
            x=int(round(pt[0]/self.down_sample_rate))
            y=int(round(pt[1]/self.down_sample_rate))
            try:
                post_dotmap[x,y] += 1.  # 一副图片只有这个点为1
            except Exception:
                post_dotmap[x-1,y-1] = 1.
        return post_dotmap







