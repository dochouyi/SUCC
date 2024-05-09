import numpy as np
from sklearn.metrics import r2_score





def project_points(p1,p2,p3):
    '''
    计算点p3投影到直线的投影点坐标
    :param p1: 线段端点1，均是np.array([x,y])的形式
    :param p2: 线段端点2
    :param p3: 要投影的点
    :return:返回投影点
    '''
    l2 = np.sum((p1-p2)**2)
    t = max(0, min(1, np.sum((p3 - p1) * (p2 - p1)) / l2))
    project_point = p1 + t * (p2 - p1)
    return project_point

def fit_curve_and_points(points,r2_thred=0.6):
    '''
    输入要经过拟合的点，如果拟合的点经过线性拟合后r2 score小于阈值，则返回False,none，否则返回True,经过矫正的点[[x,y]]
    :param points:
    :param r2_thred:float
    :return:
    '''
    points=np.array(points)
    x=points[:,0]
    y=points[:,1]
    model = np.polyfit(x, y, 1)
    predict = np.poly1d(model)
    score_value=r2_score(y, predict(x))
    if score_value>r2_thred:
        x1=min(x)-100
        x2=max(x)+100
        y1=predict(x1)
        y2=predict(x2)
        fit_points=[]
        for point in points:
            po=project_points(np.array([x1,y1]),np.array([x2,y2]),point)
            po=[round(po[0]),round(po[1])]
            fit_points.append(po)
        return True,fit_points
    else:
        return False,None


