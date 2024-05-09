import scipy
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.spatial



class Visualize_class:

    def __init__(self,load_gt_func,max_sigma,sigma_rate):
        '''
        :param prepare_class: 这个是输入的预处理类，例如ShhaiPrepare的实例
        '''
        self.load_gt_func=load_gt_func
        self.max_sigma = max_sigma
        self.sigma_rate = sigma_rate

    def draw_head_dot(self,image_file, gt_file, radius=5, show=True):
        '''
        在原图上画圆形实心点
        :param image_file:图像的路径
        :param gt_file:对应的gt的路径
        :param radius:默认即可，是圆圈的半径
        :param show:是否默认显示
        :return:返回融合之后的PIL格式的图像
        '''
        img = Image.open(image_file, mode='r').convert("RGB")
        locations = self.load_gt_func(gt_file)
        gt = Image.new("RGB", img.size, "blue")
        drawObject = ImageDraw.Draw(gt)
        for i in locations:
            drawObject.ellipse((i[0], i[1], i[0] + radius, i[1] + radius), fill="red")  # 在image上画一个红色的圆
        result_img = Image.blend(img, gt, 0.3)
        if show:
            result_img.show()
        return result_img


    @staticmethod
    def draw_density_dot_heatmap(image,density_map,dot_map):
        '''
        三张图并排显示
        :param image:ndarray 2维的numpy格式的原图
        :param density_map:ndarray
        :param dot_map:ndarray
        :return:
        '''
        img = Image.fromarray(image.astype('uint8')).convert('RGB')
        plt.figure('show')
        plt.subplot(131)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(132)
        cmap = cm.get_cmap('jet',1000)
        plt.imshow(density_map, cmap=cmap, extent=None, aspect='equal')
        plt.axis('off')
        sum=np.sum(density_map)
        print('density_count:',sum)
        plt.subplot(133)
        plt.imshow(dot_map,extent=None, aspect='equal')
        plt.axis('off')
        sum = np.sum(dot_map)/255
        print('true_count:', sum)
        plt.show()


    def adaptive_radius(self, locations):
        '''
        这是一个工具函数，供draw_head_adaptive_theta_circle调用
        :param locations:输入是2维ndarray坐标点的序列
        :return:每一个人头点的半径
        '''
        leafsize = 2048
        tree = scipy.spatial.KDTree(locations.copy(), leafsize=leafsize)  # 建立kdtree
        distances, locations = tree.query(locations, k=4, eps=10.)  # 寻找每个节点与自己最近的两个节点
        radius=[]
        for i, pt in enumerate(locations):
            sigma = np.average(distances[i][1:])  # 与第i个节点距离最近的节点
            if sigma > self.max_sigma:
                sigma = self.max_sigma
            sigma = sigma * self.sigma_rate
            radius.append(sigma)
        return radius

    def draw_head_adaptive_theta_circle(self, image_file, gt_file):
        '''
        绘制每个人头的大小圆形
        :param image_file:图像的路径
        :param gt_file:gt文件的路径
        :return:返回融合之后的图
        '''
        image = Image.open(image_file, mode='r').convert("RGB")
        locations = self.load_gt_func(gt_file)
        radius=self.adaptive_radius(locations)  # 生成人头点图
        draw = Image.new("RGB", image.size, "blue")
        drawObject = ImageDraw.Draw(draw)
        for i,loc in enumerate(locations):
            drawObject.ellipse((loc[0]-radius[i]/2, loc[1]-radius[i]/2, loc[0] + radius[i]/2, loc[1] + radius[i]/2), fill="red")  # 在image上画一个红色的圆
        result = Image.blend(image, draw, 0.3)
        result.show()
        return result
