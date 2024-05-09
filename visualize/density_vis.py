import scipy
import numpy as np
from PIL import Image
from PIL import ImageDraw
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import scipy.spatial
import math
import torch
import io

def draw_tensor_density(density_maps):
    '''
    :param density_maps: numpy或者是tensor cuda或者是tensor cpu类型，可以有多个冗余维度
    :return:
    '''
    fig_num=len(density_maps)
    rows=math.ceil(fig_num/2)
    cols=2
    plt.figure('density-show')
    cmap = cm.get_cmap('jet',1000)
    for i,den_map in enumerate(density_maps):
        if torch.is_tensor(den_map):
            den_map=den_map.data.cpu().numpy()
        den_map=np.squeeze(den_map)
        plt.subplot(rows,cols,i+1)
        plt.imshow(den_map, cmap=cmap, extent=None, aspect='equal')
        plt.axis('off')
    plt.show()




def save_tensor_density(density_maps):
    '''
    :param density_maps: numpy或者是tensor cuda或者是tensor cpu类型，可以有多个冗余维度
    :return:list(PIL.Image)
    '''
    pil_imgs=[]
    for i,den_map in enumerate(density_maps):
        if torch.is_tensor(den_map):
            den_map=den_map.data.cpu().numpy()
        den_map=np.squeeze(den_map)
        plt.figure('density-show')
        cmap = cm.get_cmap('jet',1000)
        plt.subplot(1,1,1)
        x=den_map.shape[0]-10
        y=20
        num=float(np.sum(den_map))
        plt.text(y, x,"{name}->:{num}".format(name=i,num=round(num,1)), fontsize=12,color = "r")
        plt.axis('off')
        plt.imshow(den_map, cmap=cmap, extent=None, aspect='equal')
        buf = io.BytesIO()
        plt.savefig(buf,format='png',bbox_inches='tight',pad_inches=0.0)
        buf.seek(0)
        im = Image.open(buf)
        buf.close()
        plt.cla()
        pil_imgs.append(im)

    return pil_imgs


