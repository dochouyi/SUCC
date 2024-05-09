import torch.utils.data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from datasets.imgDataset import ImgDataset
from avcc_utils.video_stream import Video_Stream
import numpy as np



class AVCC:
    def __init__(self):
        self.queue_points = list()#存储所有的点集合，经过处理的点的集合
        self.queue_pil_imgs = list()#存储所有要处理的pil格式文件，这个与queue_points是一一对应的
        self.queue_size=50#大小限制
        self.first_flag=True
        self.link_list=[]#一共当前点数条边，每条边的内容是往前延伸的
        self.save_index=0
        self.coodi_file="./g2_img/data"

    #输入5个密度图，将对应的值相机，然后用漫水填充，然后聚类，每个聚类的大小有限制不超过一个数额，取每个聚类中心点坐标，高斯核根据点的y的值以一定的比例调整
    #返回一系列2维点坐标
    def density_maps_process(self,density_maps,flow_map_dens,pil_img,index,flow_mag_thred=0.5):
        '''
        :param density_maps:list(Tensor(1,1,128,72))
        :param flow_map_dens:list(Tensor(1,2,128,72))
        :param pil_img:list(Pil.Image*5)
        :param index:int
        :return:
        '''
        maps=[]
        for i,den_map in enumerate(density_maps):
            den_map=np.squeeze(den_map.data.cpu().numpy())
            maps.append(den_map)

        mid_i=len(flow_map_dens)//2
        stack_map=0

        flow_map1=abs(np.squeeze(flow_map_dens[mid_i-2].data.cpu().numpy())[0])
        flow_map2=abs(np.squeeze(flow_map_dens[mid_i+2].data.cpu().numpy())[0])
        flow_map1[np.abs(flow_map1) < flow_mag_thred] = 0
        flow_map2[np.abs(flow_map1) < flow_mag_thred] = 0
        flow_map1[np.abs(flow_map1) > flow_mag_thred] = 1
        flow_map2[np.abs(flow_map1) > flow_mag_thred] = 1
        for i,den_map in enumerate(maps):
            if i<=mid_i:
                stack_map+=maps[i]*flow_map1
            else:
                stack_map+=maps[i]*flow_map2

        peak_list,value_list=self.get_local_peaks(stack_map)
        peak_list,value_list=self.__remove_close_points(peak_list,value_list,min_dis=10)
        peak_list=np.array(peak_list)*8
        # img=draw_head_dot(pil_img[mid_i],peak_list,show=False)
        #
        # img.save("./points_pic/{index}.png".format(index=index))
        self.queue_points.append(peak_list)
        self.queue_pil_imgs.append(pil_img[mid_i])
        if len(self.queue_points)>self.queue_size:
            self.queue_points.pop(0)
            self.queue_pil_imgs.pop(0)

        if self.first_flag==True:
            self.first_flag=False
        else:
            match_results=self.bi_graph_match(self.queue_points[-2],self.queue_points[-1])#两帧之间的匹配边
            self.link_list=self.link_adj(self.link_list,match_results)#计算累计的匹配边
            show_points=self.process(self.link_list,self.queue_points)

            # img=draw_head_dot(self.queue_pil_imgs[-1],show_points,show=False)
            # img.save("./points_pic/{index}.png".format(index=index))
    def fit_curve(self, link,points_list,queue_pil_imgs):
        '''
        拟合点的具体函数操作，需要注意当前图像已经加入到队列中，所以要先排除掉当前图像，因为进行这一步的必要条件是当前链路的连接刚好在此时断开
        :param link:
        :return:
        '''
        link_points=[]
        for i in range(len(link)):#每条边中的每个元素是
            index=link[-1-i]#从最后一个元素开始
            p=points_list[-2-i][index]#从倒数第二个元素开始
            link_points.append(p)
        #具体的线性拟合操作
        #生成修正之后的每一个点
        flag,fit_points=fit_curve_and_points(link_points)
        if flag==True:
            ps_figure(self.queue_pil_imgs[-1-len(fit_points):-1],fit_points)


    def ps_figure(self, pil_imgs_list,fit_points):
        d=[50,100,0.2]#分别是要截取出来的图像的宽度，高度和头部点距离上边的占比


        for i,im in enumerate(pil_imgs_list):
            if not i==len(pil_imgs_list)//2:
                continue
            point=fit_points[i]

            left = round(point[1]-0.5*d[0])
            top = round(point[0]-d[2]*d[1])
            right = round(left+d[0])
            bottom = round(top+d[1])
            im1 = im.crop((left, top, right, bottom))

            # im1.show()
            im1.save("g2_img/{id}.jpg".format(id=self.save_index))
            with open("{name}.txt".format(name=self.coodi_file), "a") as f:
                f.writelines("{x},{y}\n".format(x=point[0],y=point[1]))
            print(self.save_index)
            self.save_index+=1

            # draw_head_dot(im,[point],show=True)

            pass
    def process(self,link_list,points_list,l_thred=4):
        '''
        这里是已有的所有边，还有缓存的逐帧的点的信息，第三个参数是连续多少个成功跟踪算作一个，返回的是每一张图中需要绘制的当前和历史的所有点
        :param link_list: 这里的是以当前帧为基准，前面各个帧到当前帧的点的序号的数组的数组
        :param points_list: 最后一个元素是当前的所有点，大小是绝对够的，从后往前找补
        :param l_thred: 长度的阈值限制
        :return:
        '''
        new_link_list=[]
        for link in link_list:#所有的节点的长度要求不小于阈值
            if len(link)>=l_thred:
                new_link_list.append(link)
        show_points=[]
        for link in new_link_list:#所有的边
            link_points=[]
            for i in range(len(link)):#每条边中的每个元素是
                try:
                    index=link[-1-i]
                    p=points_list[-1-i][index]
                    link_points.append(p)
                except Exception:
                    pass
            print(link_points)
            # show_points.append(link_points[0])
            show_points.extend(link_points)
        return show_points


    #输入连续的5张图像，先推理出要得到的密度图
    #imgs是list(PIL.Image)
    def inference(self,imgs,model,scale_times,index):
        dataset_seq=ImgDataset(imgs)
        data_loader = DataLoader(dataset_seq, batch_size=len(imgs),num_workers=5)



        for i,img in enumerate(data_loader):
            with torch.no_grad():
                img = Variable(img).cuda()
                estDensityMap= model(img)
                estDensityMaps=estDensityMap*(1.0/scale_times)

                flow_map_dens=self.calcu_optical_flow(img,flow_net)
                warped_density_maps=self.warp_density_maps(flow_map_dens,estDensityMaps)
                self.density_maps_process(warped_density_maps,flow_map_dens,imgs,index)


    def infer_test(self,model,scale_times):
        vs=Video_Stream()
        i=0
        while(True):
            flag,imgs=vs.read()
            self.inference(imgs,model,scale_times,i)
            i=i+1
