import numpy as np
import math




def bi_graph_match(points_pre,points_cur,value_thred=15):
    '''
    两个队列的点集进行匹配，返回成功匹配的index匹配[[0,1],[2,6]]表示前一个点集中的点0和后一个点集中的点1是匹配上的
    这个是计算出来每一帧的点之后首先要做的，目的是生成边
    :param points_pre:list([x,y])
    :param points_cur:list([x,y])
    :param value_thred:int
    :return:
    '''
    if len(points_pre)==0 or len(points_cur)==0:
        return []
    dis_map=np.zeros((len(points_pre),len(points_cur)),dtype= np.float32)
    for i,point_pre in enumerate(points_pre):
        for j,point_cur in enumerate(points_cur):
            dis2=(point_pre[0]-point_cur[0])*(point_pre[0]-point_cur[0])+(point_pre[1]-point_cur[1])*(point_pre[1]-point_cur[1])
            dis=math.sqrt(dis2)
            dis_map[i,j]=dis
    match_results=[]
    for i,row_data in enumerate(dis_map):
        index=np.argmin(row_data,axis=0)
        if row_data[index]<=value_thred:
            match_results.append([i,index])
    return match_results

def link_adj(link_list,points_list,l_thred=4):
    '''
    第一个参数是历史积累的边信息，第二个参数是新的边
    :param link_list:list(list(2++))，list每个里面存储的是一条边，是上一个节点往前数的连接的节点对应关系
    :param points_list:list(list(2))
    :return:
    '''
    new_link_list=[]
    flags=np.zeros(len(link_list))#[0,0,0,0,0,0,0,0,0]表示经没经过添加新点的记录
    for points in points_list:
        leaf_flag=True
        for i,link in enumerate(link_list):
            if link[-1]==points[0]:
                flags[i]=1
                temp=link.copy()
                temp.append(points[1])
                new_link_list.append(temp)
                leaf_flag=False
                break
        if leaf_flag==True:
            new_link_list.append(points)
    # #历史存储的点消失的时候，对这个点历史信息要进行清算了，进行拟合处理的时候到了
    for i,flag in enumerate(flags):
        if flag==0:
            if len(link_list[i])<l_thred:
                continue
            else:
                self.fit_curve(link_list[i],self.queue_points)#对多个点进行去除异常点和直线拟合估计
    return new_link_list