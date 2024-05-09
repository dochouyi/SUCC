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

def draw_head_dot(pil_img, points, radius=5, show=True):

    gt = Image.new("RGB", pil_img.size, "blue")
    drawObject = ImageDraw.Draw(gt)
    for i in points:
        i=[i[1],i[0]]
        drawObject.ellipse((i[0], i[1], i[0] + radius, i[1] + radius), fill="red")  # 在image上画一个红色的圆
    result_img = Image.blend(pil_img, gt, 0.2)
    if show:
        result_img.show()
    return result_img

