from PIL import Image
import glob
import os
from visualize.pic_vis import draw_head_dot
import numpy as np


def inspect(point,point_set,d=[50,100,0.2]):
    '''
    判断当前目标点point和point_set中的所有点是否在位置上有冲突，有冲突的话返回False,没有冲突返回True
    :param point:
    :param point_set:
    :return:
    '''
    thred_dis=d[0]*d[0]+d[1]*d[1]
    for po in point_set:
        dis2=(po[0]-point[0])*(po[0]-point[0])+(po[1]-point[1])*(po[1]-point[1])
        if dis2<thred_dis:
            return False
    return True

def judge_finish(flag_list):
    '''
    flag_list中默认全部是0，如果存在没有变为1的位置，那么就证明没有结束
    :param flag_list:
    :return:
    '''
    for flag in flag_list:
        if flag==0:
            return False
    return True

def group_points(points,num_thred=10):
    '''
    points表明提取到的那些点，点的数量表示一共生成的图像块的数量，通过这个函数，生成不冲突的，每个图像中不超过num_thred个图像补丁的所有的点的group方案
    :param points:
    :param num_thred:
    :return:
    '''
    flag=np.zeros(len(points), dtype='uint8')
    group_list=[]
    index_list=[]
    while(judge_finish(flag)==False):
        point_set=[]
        index_set=[]
        for i,point in enumerate(points):
            if flag[i]==1:
                continue
            if inspect(point,point_set)==True:#测试距离是否满足要求
                point_set.append(point)
                index_set.append(i)
                flag[i]=1
                if len(point_set)>=num_thred:
                    break
        group_list.append(point_set)
        index_list.append(index_set)
    return group_list,index_list


def read_points(file="/home/houyi/Code/activevcc/g2_img_2/data.txt"):
    '''
    从txt文件中一行行读取图像中的人头的点的坐标
    :param file:
    :return:
    '''
    coods=[]
    with open(file, "r") as f:
        datas=f.readlines()
        for data in datas:
            data=data.strip()
            x,y=data.split(',')
            x=int(x)
            y=int(y)
            coods.append([x,y])
    return coods


def generate_gt(coods,bg_img_init,saved_path='/home/houyi/Code/activevcc/gt_data/',d=[50,100,0.2]):
    group_list,index_list=group_points(coods)
    for j,indexes in enumerate(index_list):
        coods=group_list[j]
        bg_img=bg_img_init.copy()
        for i,ind in enumerate(indexes):
            file='/home/houyi/Code/activevcc/g2_img_2/{i}.jpg'.format(i=ind)

            img = Image.open(file, mode='r').convert("RGB")
            point=coods[i]

            left = round(point[1]-0.5*d[0])
            top = round(point[0]-d[2]*d[1])
            right = round(left+d[0])
            bottom = round(top+d[1])
            box=(left,top,right,bottom)
            bg_img.paste(img,box)

        # gt_img=draw_head_dot(bg_img,coods,show=False)
        bg_img.save(os.path.join(saved_path,"images","{j}.jpg".format(j=j)))
        # gt_img.show()

        with open(os.path.join(saved_path,"ground-truth","{j}.txt".format(j=j)), "w") as f:
            for point in coods:
                f.writelines("{x},{y}\n".format(x=point[0],y=point[1]))


if __name__=="__main__":
    bg_file_path='bg.jpg'
    bg_img=Image.open(bg_file_path, mode='r').convert("RGB")
    coords=read_points()
    generate_gt(coords,bg_img)