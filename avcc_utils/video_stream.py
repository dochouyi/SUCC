import cv2
from PIL import Image



class Video_Stream:
    def __init__(self,file_name='./shopping/0.mp4',capture_fps=30,img_num=5):
        '''
        :param file_name:视频文件的路径
        :param capture_fps:目标视频的帧率每秒
        :param img_num:每一次read进入的图像的数目
        '''
        self.cap = cv2.VideoCapture(file_name)
        self.capture_fps=capture_fps
        self.img_num=img_num

    def read(self):
        '''
        :return:True,list(PIM.Image) or False,None
        '''
        i=0
        img_list=[]
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            i=i+1
            if ret == True and i%(30//self.capture_fps)==0:
                resized_img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

                img_list.append(resized_img)
                if self.img_num==len(img_list):
                    return True,img_list
            elif ret == True and i%(30//self.capture_fps)!=0:
                continue
            else:
                break
        self.cap.release()
        return False,None
