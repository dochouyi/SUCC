import numpy as np
import scipy
import scipy.spatial

def remove_close_points_one(peak_list,value_list, min_dis):
    '''
    :param peak_list: list([x,y])
    :param value_list: list(value)
    :return:
    '''
    pts=np.array(peak_list)
    new_peak_list=[]
    new_value_list=[]
    try:
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=256)
        distances, adj = tree.query(pts, k=2)

        for i,dis in enumerate(distances):
            if dis[1]<min_dis:
                adj_index=adj[i][1]
                if value_list[i]<value_list[adj_index]:
                    continue
            new_peak_list.append(peak_list[i])
            new_value_list.append(value_list[i])
        return new_peak_list,new_value_list
    except:
        return [],[]

def remove_close_points(peak_list,value_list,min_dis):
    '''
    :param peak_list: list([x,y])
    :param value_list: list(value)
    :return:
    '''
    while True:
        not_complete=False
        peak_list,value_list=remove_close_points_one(peak_list,value_list,min_dis)
        if peak_list==[]:
            break
        pts=np.array(peak_list)
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=256)
        distances, _ = tree.query(pts, k=2)
        for i,dis in enumerate(distances):
            if dis[1]<min_dis:
                not_complete=True
                break
        if not_complete==True:
            continue
        else:
            break
    return peak_list,value_list

def get_local_peaks(stack_map,min_value_thred=0.1):
    '''
    :param stack_map: ndarray(128,72)
    :param min_value_thred:float
    :return:
    '''
    stack_map_temp=np.pad(stack_map, ((1, 1), (1, 1)), 'constant', constant_values=0)
    h,w=stack_map_temp.shape
    peak_list=[]
    value_list=[]
    for i in range(h):
        if i==0 or i==h-1:
            continue
        for j in range(w):
            if j==0 or j==h-1:
                continue
            d=stack_map_temp[i][j]
            if d>stack_map_temp[i-1][j] and d>stack_map_temp[i+1][j] \
                    and d>stack_map_temp[i-1][j-1] and d>stack_map_temp[i+1][j-1] \
                    and d>stack_map_temp[i][j-1] and d>stack_map_temp[i][j+1] \
                    and d>stack_map_temp[i+1][j+1] and d>stack_map_temp[i-1][j+1] \
                    and d>min_value_thred:
                peak_list.append([i,j])
                value_list.append(stack_map_temp[i][j])
    return peak_list,value_list