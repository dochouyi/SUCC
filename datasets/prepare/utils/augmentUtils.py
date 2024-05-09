
import numpy as np
import random

# 用于单帧图像人群计数的数据增强
#######################################################################################################################
def generate_slices(image, density_map, dot_map):
    '''
    :param image(ndarray(img_height,img_width,3)):uint8
    :param density_map(ndarray(den_height,den_width)):float32
    :param dot_map(ndarray(den_height,den_width)):uint8
    :param down_rate(8):
    :return:[list:9(ndarray(img_height//2,img_width//2,3))]...
    '''
    out_images = []
    out_density_maps = []
    out_dot_maps = []
    try:
        img_h, img_w, _ = image.shape
        den_h, den_w= density_map.shape
        down_rate=img_h//den_h
    except BaseException:
        print('n-dim exception occuer!!!')
        return
    width = img_w // 2
    height = img_h // 2
    for w_index in range(2):
        for h_index in range(2):
            px = w_index * width
            py = h_index * height
            out_images.append(image[py:py + height, px:px + width])
            out_density_maps.append(density_map[py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
            out_dot_maps.append(dot_map[py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
    px = img_w // 2 - width // 2
    py = img_h // 2 - height // 2
    out_images.append(image[py:py + height, px:px + width])
    out_density_maps.append(density_map[py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
    out_dot_maps.append(dot_map[py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
    px_max = width
    py_max = height
    for w_index in range(2):
        for h_index in range(2):
            px = random.randint(0, px_max - 1)
            py = random.randint(0, py_max - 1)
            out_images.append(image[py:py + height, px:px + width])
            out_density_maps.append(density_map[py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
            out_dot_maps.append(dot_map[py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
    return (out_images, out_density_maps,out_dot_maps)



def flip_slices(images, density_maps,dot_maps):
    out_images = []
    out_density_maps = []
    out_dot_maps = []
    for i, img in enumerate(images):
        out_images.append(img)
        out_density_maps.append(density_maps[i])
        out_dot_maps.append(dot_maps[i])
        out_images.append(np.fliplr(img))
        out_density_maps.append(np.fliplr(density_maps[i]))
        out_dot_maps.append(np.fliplr(dot_maps[i]))
    return (out_images, out_density_maps,out_dot_maps)


# 用于视频人群计数的数据增强
#######################################################################################################################
def video_generate_slices(images, density_maps, dot_maps, down_rate=8):#处理一个大的视频序列
    '''
    :param images(list:seq_num(adarray(h,w,3))):连续的N帧图像
    :param density_maps(list:seq_num(adarray(h//down_rate,w//down_rate))):连续的N帧密度图（密度图和点图的大小是图像大小的8分之一）
    :param dot_maps(list:seq_num(adarray(h//down_rate,w//down_rate))):连续的N帧点图
    :return:[images(list:seq_num*9(adarray(h//2,w//2,3)))...]
    '''
    out_images = []
    out_density_maps = []
    out_dot_maps = []
    indexes=[]
    try:
        img_h, img_w, _ = images[0].shape
    except BaseException:
        print('n-dim exception occuer!!!')
        return
    width = img_w // 2
    height = img_h // 2
    mode_index=-1
    #四个角的数据增强
    for w_index in range(2):
        for h_index in range(2):
            mode_index=mode_index+1
            px = w_index * width
            py = h_index * height
            for i,img in enumerate(images):
                out_images.append(img[py:py + height, px:px + width])
                out_density_maps.append(density_maps[i][py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
                out_dot_maps.append(dot_maps[i][py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
                indexes.append([mode_index,i])
    #中间的数据增强
    mode_index=mode_index+1
    px = img_w // 2 - width // 2
    py = img_h // 2 - height // 2
    for i,img in enumerate(images):
        out_images.append(img[py:py + height, px:px + width])
        out_density_maps.append(density_maps[i][py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
        out_dot_maps.append(dot_maps[i][py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
        indexes.append([mode_index,i])
    #四个随机的数据
    px_max = width
    py_max = height
    for w_index in range(2):
        for h_index in range(2):
            mode_index=mode_index+1
            px = random.randint(0, px_max - 1)
            py = random.randint(0, py_max - 1)
            for i,img in enumerate(images):
                out_images.append(img[py:py + height, px:px + width])
                out_density_maps.append(density_maps[i][py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
                out_dot_maps.append(dot_maps[i][py//down_rate:(py + height)//down_rate, px//down_rate:(px + width)//down_rate])
                indexes.append([mode_index,i])
    return (out_images, out_density_maps,out_dot_maps,indexes)
