import torch.utils.data
import torch.nn.functional as F
from optical_flow.PWCNet import pwc_dc_net,flow_warp

#计算光流和warp的操作在这里进行
class Optical_Flow:

    def __init__(self, weights_path = './weights/pwc_net.pth.tar'):
        self.flow_net = pwc_dc_net(weights_path).cuda().eval()


    def calcu_optical_flow(self, imgs):
        '''
        只要保证imgs是连续的图像即可,imgs的数量不限
        输出面向中间的图像的所有的光流图，中间的光流图的值为1
        :param imgs: Tensor(bs,3,h,w):cuda0,torch.float32
        :param flow_net: Optical flow network
        :return list(Tensor(1,2,h//8,w//8):cuda0,torch.float32)
        '''
        img_height=imgs.shape[2]
        img_width=imgs.shape[3]
        imgs = torch.unsqueeze(imgs, dim=1)
        of_num=len(imgs)//2
        mid_i=of_num
        flow_map_den_list=[]
        for i in range(len(imgs)):
            if i==mid_i:
                flow_map_den_list.append(1)
            else:
                pair = torch.cat((imgs[mid_i], imgs[i]), dim=1)
                flow_map = self.flow_net(pair)*2.5
                flow_map_den = F.interpolate(flow_map, size=(img_height//8,img_width//8), mode="bilinear")
                flow_map_den_list.append(flow_map_den)
        return flow_map_den_list



    def warp_density_maps(self, flow_map_dens, density_maps):
        '''
        输入四个光流图，还有四个密度图，输出4个分别warp之后的密度图，中间的是原密度图
        :param flow_den_maps:list(Tensor(1,2,128,72))
        :param density_maps:Tensor(5,1,128,72)
        :return:list(Tensor(1,1,128,72))
        '''
        density_maps = torch.unsqueeze(density_maps, dim=1)
        warped_imgs=[]
        mid_i=len(flow_map_dens)//2
        for i,flow_map in enumerate(flow_map_dens):
            if i==mid_i:
                warped_imgs.append(density_maps[i])
            else:
                warped=flow_warp(density_maps[i], flow_map_dens[i])
                warped_imgs.append(warped)
        return warped_imgs